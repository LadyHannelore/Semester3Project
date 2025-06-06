// Performance constants
const DEBOUNCE_DELAY = 300; // ms for search input
const THROTTLE_DELAY = 16; // ms for 60fps animations
const CACHE_DURATION = 300000; // 5 minutes
const MAX_VISIBLE_ROWS = 100; // Virtual scrolling threshold

// Utility functions for performance optimization
function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

function throttle(func, delay) {
    let lastCall = 0;
    return function (...args) {
        const now = new Date().getTime();
        if (now - lastCall < delay) return;
        lastCall = now;
        return func.apply(this, args);
    };
}

// Cache for filtered results
const filterCache = new Map();

// Global state
let currentDataset = null;
let filteredData = [];
let currentPage = 1;
const rowsPerPage = 25;
let sortColumn = null;
let sortDirection = 'asc';
let flaggedStudents = new Set();
let currentProfileStudentId = null; // Added for reliable flagging

// Add initialization function to maintain proper structure
document.addEventListener('DOMContentLoaded', function() {
    // Initialize student explorer functionality
    loadData();
    initializeEventListeners();
});

// Load data from localStorage or API
function loadData() {
    loadDataset(); // <-- Actually load the dataset from localStorage
}

// Load dataset from localStorage
function loadDataset() {
    const stored = localStorage.getItem('classforgeDataset');
    if (stored) {
        currentDataset = JSON.parse(stored);
        if (!currentDataset.headers || !currentDataset.rows) {
            showEmptyState("Invalid dataset structure in localStorage.");
            currentDataset = null; // Prevent further errors
            return;
        }
        filteredData = [...currentDataset.rows];
        renderTableHeaders();
        initializeDynamicFilters();
        updateTable();
    } else {
        showEmptyState("No dataset loaded. Please upload or generate data first.");
    }
}

// Show empty state with a professional message and disables all controls
function showEmptyState(message = "No dataset loaded.") {
    const tbody = document.getElementById('studentsTableBody') || document.querySelector('tbody');
    if (!tbody) return;
    const emptyRow = document.createElement('tr');
    const emptyCell = document.createElement('td');
    emptyCell.colSpan = 100;
    emptyCell.innerHTML = `
        <div class="empty-state-message">
            <p>${message}</p>
            <p>Please go to the Upload & Simulate page to:</p>
            <ul>
                <li>📤 Upload a CSV file with student data</li>
                <li>🔄 Or generate synthetic data for testing</li>
            </ul>
            <button class="action-btn" onclick="window.location.href='upload.html'">
                Go to Upload Page
            </button>
        </div>
    `;
    emptyRow.appendChild(emptyCell);
    tbody.innerHTML = '';
    tbody.appendChild(emptyRow);

    // Disable controls
    ['searchInput', 'bullyingFilter', 'wellbeingFilter', 'prevPage', 'nextPage'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = true;
    });
    const dynamicFiltersContainer = document.getElementById('dynamicFiltersContainer');
    if (dynamicFiltersContainer) {
        dynamicFiltersContainer.querySelectorAll('input, select').forEach(el => el.disabled = true);
    }
    // Reset pagination info
    ['startRow', 'endRow', 'totalRows', 'currentPage'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = id === 'currentPage' ? 'Page 0' : '0';
    });
}

// Initialize event listeners with optimized handlers
function initializeEventListeners() {
    // Search input with debouncing
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', debounce(handleSearch, DEBOUNCE_DELAY));
    }

    // Filter selects with throttling
    const bullyingFilter = document.getElementById('bullyingFilter');
    if (bullyingFilter) {
        bullyingFilter.addEventListener('change', throttle(applyFilters, THROTTLE_DELAY));
    }
    
    const wellbeingFilter = document.getElementById('wellbeingFilter');
    if (wellbeingFilter) {
        wellbeingFilter.addEventListener('change', throttle(applyFilters, THROTTLE_DELAY));
    }

    // Pagination buttons
    const prevButton = document.getElementById('prevPage');
    const nextButton = document.getElementById('nextPage');
    if (prevButton) {
        prevButton.addEventListener('click', () => changePage(currentPage - 1));
    }
    if (nextButton) {
        nextButton.addEventListener('click', () => changePage(currentPage + 1));
    }

    // Sort headers (will be attached in renderTableHeaders)
}

// Render table headers dynamically and attach sort event listeners
function renderTableHeaders() {
    if (!currentDataset || !currentDataset.headers) return;
    const tableHead = document.querySelector('.data-table thead');
    tableHead.innerHTML = '';
    const headerRow = document.createElement('tr');
    currentDataset.headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText.replace(/_/g, ' ');
        th.dataset.sort = headerText;
        th.classList.add('sortable');
        const sortIcon = document.createElement('span');
        sortIcon.className = 'sort-icon';
        sortIcon.innerHTML = ' ↕️';
        th.appendChild(sortIcon);
        th.tabIndex = 0;
        th.setAttribute('role', 'button');
        th.setAttribute('aria-label', `Sort by ${headerText.replace(/_/g, ' ')}`);
        th.addEventListener('click', () => handleSort(th.dataset.sort));
        headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);
}

// Initialize dynamic filters based on dataset
function initializeDynamicFilters() {
    if (!currentDataset || !currentDataset.headers) return;

    const container = document.getElementById('dynamicFiltersContainer');
    container.innerHTML = ''; // Clear existing dynamic filters

    // Example: Add a filter for K6_Score if it exists
    const k6Header = 'K6_Score';
    if (currentDataset.headers.includes(k6Header)) {
        const k6Label = document.createElement('label');
        k6Label.textContent = 'K6 Score:';
        k6Label.htmlFor = 'k6Filter';
        
        const k6Select = document.createElement('select');
        k6Select.id = 'k6Filter';
        k6Select.className = 'filter-select';
        k6Select.innerHTML = `
            <option value="">All</option>
            <option value="low">Low (0-10)</option>
            <option value="medium">Medium (11-17)</option>
            <option value="high">High (18-24)</option>
        `;
        k6Select.addEventListener('change', applyFilters);
        
        container.appendChild(k6Label);
        container.appendChild(k6Select);
    }
    // Add more dynamic filters here for other columns as needed
}

// Handle search with caching
function handleSearch(e) {
    if (!currentDataset) return;
    const searchTerm = e.target.value.toLowerCase();
    
    // Check cache first
    const cacheKey = `search_${searchTerm}`;
    if (filterCache.has(cacheKey)) {
        const cachedResult = filterCache.get(cacheKey);
        if (Date.now() - cachedResult.timestamp < CACHE_DURATION) {
            filteredData = cachedResult.data;
            currentPage = 1;
            applyFilters();
            return;
        }
    }
    
    // Filter based on StudentID (assuming it's the first column) or any other identifiable column
    const studentIdIndex = currentDataset.headers.indexOf('StudentID'); // Changed from 'Student_ID'
    let searchResult;
    
    if (studentIdIndex !== -1) {
        searchResult = currentDataset.rows.filter(row =>
            row[studentIdIndex] && String(row[studentIdIndex]).toLowerCase().includes(searchTerm)
        );
    } else { // Fallback if StudentID is not found, search all columns
        searchResult = currentDataset.rows.filter(row =>
            row.some(cell => cell && String(cell).toLowerCase().includes(searchTerm))
        );
    }
    
    // Cache the result
    filterCache.set(cacheKey, {
        data: searchResult,
        timestamp: Date.now()
    });
    
    filteredData = searchResult;
    currentPage = 1;
    applyFilters();
}

// Apply filters with caching and optimization
function applyFilters() {
    if (!currentDataset) return;

    const bullyingFilter = document.getElementById('bullyingFilter').value;
    const wellbeingFilter = document.getElementById('wellbeingFilter').value;
    
    // Get dynamic filter values
    const k6FilterSelect = document.getElementById('k6Filter');
    const k6Filter = k6FilterSelect ? k6FilterSelect.value : "";
    
    // Create cache key from filter combination
    const cacheKey = `filters_${bullyingFilter}_${wellbeingFilter}_${k6Filter}`;
    
    // Check cache first
    if (filterCache.has(cacheKey)) {
        const cachedResult = filterCache.get(cacheKey);
        if (Date.now() - cachedResult.timestamp < CACHE_DURATION) {
            filteredData = cachedResult.data;
            if (sortColumn) {
                sortData(sortColumn, sortDirection);
            }
            updateTable();
            return;
        }
    }

    // Pre-calculate column indices for better performance
    const bullyingScoreIndex = currentDataset.headers.indexOf('Bullying_Score');
    const wellbeingScoreIndex = currentDataset.headers.indexOf('Wellbeing_Score');
    const k6ScoreIndex = currentDataset.headers.indexOf('K6_Score');

    // Use more efficient filtering with early returns
    const result = currentDataset.rows.filter(row => {
        // Ensure indices are valid before accessing row data
        if (bullyingFilter && bullyingScoreIndex !== -1) {
            const bullyingScore = parseInt(row[bullyingScoreIndex]);
            if (isNaN(bullyingScore)) return false;
            
            switch (bullyingFilter) {
                case 'high': if (bullyingScore < 6) return false; break;
                case 'medium': if (bullyingScore < 3 || bullyingScore > 5) return false; break;
                case 'low': if (bullyingScore > 2) return false; break;
            }
        }

        if (wellbeingFilter && wellbeingScoreIndex !== -1) {
            const wellbeingScore = parseInt(row[wellbeingScoreIndex]);
            if (isNaN(wellbeingScore)) return false;
            
            switch (wellbeingFilter) {
                case 'high': if (wellbeingScore <= 7) return false; break;
                case 'medium': if (wellbeingScore < 3 || wellbeingScore > 7) return false; break;
                case 'low': if (wellbeingScore >= 3) return false; break;
            }
        }

        // Apply K6_Score filter (example dynamic filter)
        if (k6Filter && k6ScoreIndex !== -1) {
            const k6Score = parseInt(row[k6ScoreIndex]);
            if (!isNaN(k6Score)) {
                switch (k6Filter) {
                    case 'low': if (k6Score > 10) return false; break;
                    case 'medium': if (k6Score < 11 || k6Score > 17) return false; break;
                    case 'high': if (k6Score < 18) return false; break;
                }
            }
        }

        return true;
    });

    // Cache the result
    filterCache.set(cacheKey, {
        data: result,
        timestamp: Date.now()
    });
    
    filteredData = result;

    if (sortColumn) {
        sortData(sortColumn, sortDirection);
    }

    updateTable();
}

// Handle sorting
function handleSort(column) {
    if (!currentDataset) return;

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
    if (!currentDataset || !currentDataset.headers) return;

    const columnIndex = currentDataset.headers.indexOf(column);
    if (columnIndex === -1) return; // Column not found

    filteredData.sort((a, b) => {
        let valueA = a[columnIndex];
        let valueB = b[columnIndex];

        // Attempt to parse as numbers
        const numA = parseFloat(valueA);
        const numB = parseFloat(valueB);

        if (!isNaN(numA) && !isNaN(numB)) {
            valueA = numA;
            valueB = numB;
        } else {
            // Fallback to string comparison if not both are numbers
            valueA = String(valueA || "").toLowerCase(); // Handle null/undefined
            valueB = String(valueB || "").toLowerCase(); // Handle null/undefined
        }

        let comparison = 0;
        if (valueA > valueB) {
            comparison = 1;
        } else if (valueA < valueB) {
            comparison = -1;
        }

        return direction === 'asc' ? comparison : comparison * -1;
    });
}

// Update table with paginated and filtered data, add accessibility improvements
function updateTable() {
    if (!currentDataset || !currentDataset.headers) {
        showEmptyState("Dataset headers are missing or invalid.");
        return;
    }
    const tbody = document.getElementById('studentTableBody');
    tbody.innerHTML = '';
    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    const paginatedData = filteredData.slice(start, end);

    paginatedData.forEach(rowData => {
        const tr = document.createElement('tr');
        tr.setAttribute('tabindex', '0');
        tr.setAttribute('role', 'row');
        currentDataset.headers.forEach(header => {
            const td = document.createElement('td');
            const cellIndex = currentDataset.headers.indexOf(header);
            td.textContent = rowData[cellIndex] !== undefined ? rowData[cellIndex] : '';
            tr.appendChild(td);
        });
        // Risk highlighting
        const bullyingScoreIndex = currentDataset.headers.indexOf('Bullying_Score');
        const wellbeingScoreIndex = currentDataset.headers.indexOf('Wellbeing_Score');
        if (bullyingScoreIndex !== -1) {
            const bullyingScore = parseInt(rowData[bullyingScoreIndex]);
            if (bullyingScore >= 6) tr.classList.add('high-risk');
        }
        if (wellbeingScoreIndex !== -1) {
            const wellbeingScore = parseInt(rowData[wellbeingScoreIndex]);
            if (wellbeingScore < 3) tr.classList.add('low-wellbeing');
        }
        tr.addEventListener('click', () => showProfile(rowData));
        tbody.appendChild(tr);
    });
    updatePagination();
}

// Update pagination
function updatePagination() {
    const totalRows = filteredData.length;
    const totalPages = Math.max(1, Math.ceil(totalRows / rowsPerPage));
    
    // Clamp currentPage to valid range
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startRow = totalRows === 0 ? 0 : (currentPage - 1) * rowsPerPage + 1;
    const endRow = Math.min(currentPage * rowsPerPage, totalRows);

    document.getElementById('startRow').textContent = startRow;
    document.getElementById('endRow').textContent = endRow;
    document.getElementById('totalRows').textContent = totalRows;
    document.getElementById('currentPage').textContent = `Page ${currentPage}`;
    
    document.getElementById('prevPage').disabled = currentPage === 1;
    document.getElementById('nextPage').disabled = currentPage === totalPages || totalRows === 0;
}

// Change page
function changePage(newPage) {
    const totalRows = filteredData.length;
    const totalPages = Math.max(1, Math.ceil(totalRows / rowsPerPage));
    if (newPage < 1 || newPage > totalPages) return;
    currentPage = newPage;
    updateTable();
}

// Show student profile
function showProfile(studentData) {
    if (!currentDataset || !currentDataset.headers) return;

    const panel = document.getElementById('profilePanel');
    panel.classList.add('active');

    const studentIdIndex = currentDataset.headers.indexOf('StudentID'); // Changed from 'Student_ID'
    // Set currentProfileStudentId using StudentID if available, otherwise fallback to first column
    currentProfileStudentId = studentIdIndex !== -1 ? String(studentData[studentIdIndex]) : String(studentData[0]); 

    const profileDetailsContainer = document.getElementById('profileDetailsContainer');
    profileDetailsContainer.innerHTML = ''; // Clear previous details

    // Dynamically display all student data
    currentDataset.headers.forEach((header, index) => {
        const detailCard = document.createElement('div');
        detailCard.className = 'profile-detail-card';
        
        const title = document.createElement('h5');
        title.textContent = header.replace(/_/g, ' '); // Make header readable
        
        const value = document.createElement('div');
        value.className = 'value';
        value.textContent = studentData[index] !== undefined ? studentData[index] : '-';
        
        detailCard.appendChild(title);
        detailCard.appendChild(value);
        profileDetailsContainer.appendChild(detailCard);
    });

    // Update flag button
    const flagButton = document.getElementById('flagButton');
    const isFlagged = flaggedStudents.has(currentProfileStudentId); // Use the reliably set currentProfileStudentId
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
    if (!currentProfileStudentId) {
        console.warn("No student profile selected to flag/unflag.");
        return; 
    }

    const studentId = currentProfileStudentId; // Use the stored ID
    if (flaggedStudents.has(studentId)) {
        flaggedStudents.delete(studentId);
    } else {
        flaggedStudents.add(studentId);
    }
    
    const flagButton = document.getElementById('flagButton');
    flagButton.classList.toggle('flagged');
    flagButton.textContent = flagButton.classList.contains('flagged') ? 'Remove Flag' : 'Flag for Review';
}

// Create radar chart for student profile, with robust error handling
function createRadarChart(studentData) {
    if (!currentDataset || !currentDataset.headers) return;
    const ctx = document.getElementById('profileRadarChart').getContext('2d');
    if (window.radarChart) {
        window.radarChart.destroy();
        window.radarChart = null;
    }
    const academicHeader = 'Academic_Performance';
    const wellbeingHeader = 'Wellbeing_Score';
    const bullyingHeader = 'Bullying_Score';
    const radarContainer = document.getElementById('profileRadarChart').parentElement;
    const existingMessage = radarContainer.querySelector('p');
    if (existingMessage) existingMessage.remove();
    if (!currentDataset.headers.includes(academicHeader) ||
        !currentDataset.headers.includes(wellbeingHeader) ||
        !currentDataset.headers.includes(bullyingHeader)) {
        radarContainer.insertAdjacentHTML('afterbegin', '<p class="radar-warning">Radar chart data incomplete (missing required columns).</p>');
        return;
    }
    const academicScoreIndex = currentDataset.headers.indexOf(academicHeader);
    const wellbeingScoreIndex = currentDataset.headers.indexOf(wellbeingHeader);
    const bullyingScoreIndex = currentDataset.headers.indexOf(bullyingHeader);
    const academicScore = parseInt(studentData[academicScoreIndex]);
    const wellbeingScore = parseInt(studentData[wellbeingScoreIndex]);
    const bullyingScore = parseInt(studentData[bullyingScoreIndex]);
    if (isNaN(academicScore) || isNaN(wellbeingScore) || isNaN(bullyingScore)) {
        radarContainer.insertAdjacentHTML('afterbegin', '<p class="radar-warning">Radar chart data invalid.</p>');
        return;
    }
    window.radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Academic', 'Wellbeing', 'Bullying'], // Consider making these more descriptive if space allows
            datasets: [{
                label: 'Student Profile',
                data: [
                    academicScore / 100,
                    wellbeingScore / 10,
                    bullyingScore / 10
                ],
                backgroundColor: 'rgba(66, 153, 225, 0.3)', // Slightly more opaque
                borderColor: 'rgba(66, 153, 225, 1)',
                pointBackgroundColor: 'rgba(66, 153, 225, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(66, 153, 225, 1)',
                borderWidth: 1.5, // Slightly thicker border
                pointRadius: 4, // Slightly larger points
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Important for respecting canvas height
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    angleLines: {
                        color: 'rgba(0, 0, 0, 0.15)' // More visible angle lines
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.15)' // More visible grid lines
                    },
                    pointLabels: {
                        font: {
                            size: 16, // Increased font size for point labels (Academic, Wellbeing, etc.)
                            weight: '500'
                        },
                        color: '#333' // Darker point labels
                    },
                    ticks: {
                        backdropColor: 'rgba(255, 255, 255, 0.75)', // Background for tick labels
                        color: '#444', // Darker tick labels
                        font: {
                            size: 14 // Increased font size for radial ticks (0, 0.2, 0.4...)
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        font: {
                            size: 16 // Increased font size for legend
                        },
                        color: '#333'
                    }
                },
                title: { // Added chart title
                    display: true,
                    text: 'Student Profile Radar',
                    font: { size: 20, weight: 'bold' }, // Increased font size for chart title
                    color: '#333',
                    padding: { top: 10, bottom: 20 }
                }
            }
        }
    });
}

// Export filtered data
function exportFilteredData() {
    if (!currentDataset || !filteredData.length) {
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