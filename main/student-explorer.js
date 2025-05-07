// Global state
let currentDataset = null;
let filteredData = [];
let currentPage = 1;
const rowsPerPage = 25;
let sortColumn = null;
let sortDirection = 'asc';
let flaggedStudents = new Set();

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadDataset();
    initializeEventListeners();
});

// Load dataset from localStorage
function loadDataset() {
    const stored = localStorage.getItem('classforgeDataset');
    if (stored) {
        currentDataset = JSON.parse(stored);
        filteredData = [...currentDataset.rows];
        updateTable();
    } else {
        // Show empty state instead of redirecting
        showEmptyState();
    }
}

// Show empty state
function showEmptyState() {
    const tbody = document.getElementById('studentTableBody');
    const emptyRow = document.createElement('tr');
    const emptyCell = document.createElement('td');
    emptyCell.colSpan = 8; // Span all columns
    emptyCell.style.textAlign = 'center';
    emptyCell.style.padding = '2rem';
    emptyCell.innerHTML = `
        <div style="color: #718096; font-style: italic;">
            <p>No dataset loaded at the moment.</p>
            <p>Please go to the Upload & Simulate page to:</p>
            <ul style="list-style: none; padding: 0;">
                <li>📤 Upload a CSV file with student data</li>
                <li>🔄 Or generate synthetic data for testing</li>
            </ul>
            <button onclick="window.location.href='upload.html'" 
                    style="padding: 0.5rem 1rem; 
                           background-color: #4299e1; 
                           color: white; 
                           border: none; 
                           border-radius: 0.375rem; 
                           cursor: pointer; 
                           margin-top: 1rem;">
                Go to Upload Page
            </button>
        </div>
    `;
    emptyRow.appendChild(emptyCell);
    tbody.innerHTML = '';
    tbody.appendChild(emptyRow);

    // Disable filters and pagination
    document.getElementById('searchInput').disabled = true;
    document.getElementById('bullyingFilter').disabled = true;
    document.getElementById('wellbeingFilter').disabled = true;
    document.getElementById('prevPage').disabled = true;
    document.getElementById('nextPage').disabled = true;

    // Update pagination info
    document.getElementById('startRow').textContent = '0';
    document.getElementById('endRow').textContent = '0';
    document.getElementById('totalRows').textContent = '0';
    document.getElementById('currentPage').textContent = 'Page 0';
}

// Initialize event listeners
function initializeEventListeners() {
    // Search input
    document.getElementById('searchInput').addEventListener('input', handleSearch);

    // Filter selects
    document.getElementById('bullyingFilter').addEventListener('change', applyFilters);
    document.getElementById('wellbeingFilter').addEventListener('change', applyFilters);

    // Pagination buttons
    document.getElementById('prevPage').addEventListener('click', () => changePage(currentPage - 1));
    document.getElementById('nextPage').addEventListener('click', () => changePage(currentPage + 1));

    // Sort headers
    document.querySelectorAll('.data-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => handleSort(th.dataset.sort));
    });
}

// Handle search
function handleSearch(e) {
    const searchTerm = e.target.value.toLowerCase();
    filteredData = currentDataset.rows.filter(row => 
        row[0].toLowerCase().includes(searchTerm)
    );
    currentPage = 1;
    applyFilters();
}

// Apply filters
function applyFilters() {
    const bullyingFilter = document.getElementById('bullyingFilter').value;
    const wellbeingFilter = document.getElementById('wellbeingFilter').value;

    filteredData = currentDataset.rows.filter(row => {
        const bullyingScore = parseInt(row[currentDataset.headers.indexOf('Bullying_Score')]);
        const wellbeingScore = parseInt(row[currentDataset.headers.indexOf('Wellbeing_Score')]);

        // Apply bullying filter
        if (bullyingFilter) {
            if (bullyingFilter === 'high' && bullyingScore < 6) return false;
            if (bullyingFilter === 'medium' && (bullyingScore < 3 || bullyingScore > 5)) return false;
            if (bullyingFilter === 'low' && bullyingScore > 2) return false;
        }

        // Apply wellbeing filter
        if (wellbeingFilter) {
            if (wellbeingFilter === 'high' && wellbeingScore <= 7) return false;
            if (wellbeingFilter === 'medium' && (wellbeingScore < 3 || wellbeingScore > 7)) return false;
            if (wellbeingFilter === 'low' && wellbeingScore >= 3) return false;
        }

        return true;
    });

    if (sortColumn) {
        sortData(sortColumn, sortDirection);
    }

    updateTable();
}

// Handle sorting
function handleSort(column) {
    if (sortColumn === column) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        sortColumn = column;
        sortDirection = 'asc';
    }

    sortData(column, sortDirection);
    updateTable();
}

// Sort data
function sortData(column, direction) {
    const columnIndex = {
        'id': 0,
        'academic': currentDataset.headers.indexOf('Academic_Performance'),
        'wellbeing': currentDataset.headers.indexOf('Wellbeing_Score'),
        'bullying': currentDataset.headers.indexOf('Bullying_Score')
    }[column];

    filteredData.sort((a, b) => {
        let valueA = a[columnIndex];
        let valueB = b[columnIndex];

        // Convert to numbers for numeric columns
        if (column !== 'id') {
            valueA = parseFloat(valueA);
            valueB = parseFloat(valueB);
        }

        if (direction === 'asc') {
            return valueA > valueB ? 1 : -1;
        } else {
            return valueA < valueB ? 1 : -1;
        }
    });
}

// Update table
function updateTable() {
    const tbody = document.getElementById('studentTableBody');
    tbody.innerHTML = '';

    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const paginatedData = filteredData.slice(start, end);

    paginatedData.forEach(row => {
        const tr = document.createElement('tr');
        
        // Add row data
        row.forEach((cell, index) => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });

        // Add risk highlighting
        const bullyingScore = parseInt(row[currentDataset.headers.indexOf('Bullying_Score')]);
        const wellbeingScore = parseInt(row[currentDataset.headers.indexOf('Wellbeing_Score')]);

        if (bullyingScore >= 6) {
            tr.classList.add('high-risk');
        }
        if (wellbeingScore < 3) {
            tr.classList.add('low-wellbeing');
        }

        // Add click handler for profile
        tr.addEventListener('click', () => showProfile(row));
        tbody.appendChild(tr);
    });

    // Update pagination
    updatePagination();
}

// Update pagination
function updatePagination() {
    const totalRows = filteredData.length;
    const totalPages = Math.ceil(totalRows / rowsPerPage);
    
    document.getElementById('startRow').textContent = (currentPage - 1) * rowsPerPage + 1;
    document.getElementById('endRow').textContent = Math.min(currentPage * rowsPerPage, totalRows);
    document.getElementById('totalRows').textContent = totalRows;
    document.getElementById('currentPage').textContent = `Page ${currentPage}`;
    
    document.getElementById('prevPage').disabled = currentPage === 1;
    document.getElementById('nextPage').disabled = currentPage === totalPages;
}

// Change page
function changePage(newPage) {
    currentPage = newPage;
    updateTable();
}

// Show student profile
function showProfile(studentData) {
    const panel = document.getElementById('profilePanel');
    panel.classList.add('active');

    // Update stats
    document.getElementById('profileAcademic').textContent = 
        studentData[currentDataset.headers.indexOf('Academic_Performance')];
    document.getElementById('profileWellbeing').textContent = 
        studentData[currentDataset.headers.indexOf('Wellbeing_Score')];
    document.getElementById('profileBullying').textContent = 
        studentData[currentDataset.headers.indexOf('Bullying_Score')];

    // Update risk level
    const bullyingScore = parseInt(studentData[currentDataset.headers.indexOf('Bullying_Score')]);
    const riskLevel = bullyingScore >= 6 ? 'High' : bullyingScore >= 3 ? 'Medium' : 'Low';
    document.getElementById('profileRisk').textContent = riskLevel;

    // Update flag button
    const flagButton = document.getElementById('flagButton');
    const isFlagged = flaggedStudents.has(studentData[0]);
    flagButton.classList.toggle('flagged', isFlagged);
    flagButton.textContent = isFlagged ? 'Remove Flag' : 'Flag for Review';

    // Create radar chart
    createRadarChart(studentData);
}

// Close profile
function closeProfile() {
    document.getElementById('profilePanel').classList.remove('active');
}

// Toggle flag
function toggleFlag() {
    const studentId = document.getElementById('profileAcademic').textContent;
    if (flaggedStudents.has(studentId)) {
        flaggedStudents.delete(studentId);
    } else {
        flaggedStudents.add(studentId);
    }
    
    const flagButton = document.getElementById('flagButton');
    flagButton.classList.toggle('flagged');
    flagButton.textContent = flagButton.classList.contains('flagged') ? 'Remove Flag' : 'Flag for Review';
}

// Create radar chart
function createRadarChart(studentData) {
    const ctx = document.getElementById('profileRadarChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.radarChart) {
        window.radarChart.destroy();
    }

    const academicScore = parseInt(studentData[currentDataset.headers.indexOf('Academic_Performance')]);
    const wellbeingScore = parseInt(studentData[currentDataset.headers.indexOf('Wellbeing_Score')]);
    const bullyingScore = parseInt(studentData[currentDataset.headers.indexOf('Bullying_Score')]);

    window.radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Academic', 'Wellbeing', 'Bullying'],
            datasets: [{
                label: 'Student Profile',
                data: [
                    academicScore / 100, // Normalize to 0-1
                    wellbeingScore / 10, // Normalize to 0-1
                    bullyingScore / 10  // Normalize to 0-1
                ],
                backgroundColor: 'rgba(66, 153, 225, 0.2)',
                borderColor: 'rgba(66, 153, 225, 1)',
                pointBackgroundColor: 'rgba(66, 153, 225, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(66, 153, 225, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Export filtered data
function exportFilteredData() {
    if (!filteredData.length) {
        alert('No data to export');
        return;
    }

    const csvContent = [
        currentDataset.headers.join(','),
        ...filteredData.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'filtered_students.csv';
    link.click();
}

// Navigation function
function navigateTo(page) {
    if (page === 'overview') {
        window.location.href = 'index.html';
    } else if (page === 'allocation') {
        // Store filtered data for allocation
        localStorage.setItem('filteredStudents', JSON.stringify(filteredData));
        window.location.href = 'allocation.html';
    }
} 