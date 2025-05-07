// Global state
let allocationData = null;
let currentClassId = null;
let currentFilters = {
    academic: { min: null, max: null },
    wellbeing: { min: null, max: null },
    flags: {
        highRisk: false,
        needsAttention: false,
        bullying: false
    }
};
let sortConfig = {
    column: null,
    direction: 'asc'
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadAllocationData();
    initializeEventListeners();
});

// Load allocation data from localStorage
function loadAllocationData() {
    const stored = localStorage.getItem('allocationResults');
    if (stored) {
        allocationData = JSON.parse(stored);
        populateClassList();
        // Load first class by default
        if (allocationData.classes && allocationData.classes.length > 0) {
            loadClassDetails(0);
        }
    } else {
        showError('No allocation data found. Please run allocation first.');
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // Class search
    document.getElementById('classSearch').addEventListener('input', filterClassList);

    // Student search
    document.getElementById('studentSearch').addEventListener('input', filterStudentTable);

    // Filter button
    document.getElementById('filterBtn').addEventListener('click', showFilterModal);

    // Table sorting
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => sortTable(th.dataset.sort));
    });
}

// Populate class list
function populateClassList() {
    const classList = document.getElementById('classList');
    classList.innerHTML = '';

    allocationData.classes.forEach((classData, index) => {
        const classItem = document.createElement('div');
        classItem.className = 'class-item';
        classItem.dataset.classId = index;
        
        const hasViolations = checkViolations(classData);
        
        classItem.innerHTML = `
            <span class="class-icon">🏫</span>
            <div class="class-info">
                <div class="class-name">Class ${index}</div>
                <div class="class-stats">
                    ${classData.students.length} students
                    ${hasViolations ? '<span class="violation-indicator">⚠️</span>' : ''}
                </div>
            </div>
        `;

        classItem.addEventListener('click', () => loadClassDetails(index));
        classList.appendChild(classItem);
    });
}

// Load class details
function loadClassDetails(classId) {
    currentClassId = classId;
    const classData = allocationData.classes[classId];

    // Update active class in list
    document.querySelectorAll('.class-item').forEach(item => {
        item.classList.toggle('active', item.dataset.classId === String(classId));
    });

    // Update class header and ID
    document.getElementById('currentClassId').textContent = classId;

    // Update metrics
    updateMetrics(classData);

    // Update violation flags
    updateViolationFlags(classData);

    // Update student table
    updateStudentTable(classData.students);
}

// Update class metrics
function updateMetrics(classData) {
    const metrics = calculateClassMetrics(classData);
    
    document.getElementById('totalStudents').textContent = classData.students.length;
    document.getElementById('avgAcademic').textContent = metrics.avgAcademic.toFixed(1);
    document.getElementById('avgWellbeing').textContent = metrics.avgWellbeing.toFixed(1);
    document.getElementById('bullyingCount').textContent = metrics.bullyingCount;
}

// Calculate class metrics
function calculateClassMetrics(classData) {
    const students = classData.students;
    return {
        avgAcademic: students.reduce((sum, s) => sum + s.academicScore, 0) / students.length,
        avgWellbeing: students.reduce((sum, s) => sum + s.wellbeingScore, 0) / students.length,
        bullyingCount: students.filter(s => s.bullyingScore > 7).length
    };
}

// Check for violations
function checkViolations(classData) {
    const metrics = calculateClassMetrics(classData);
    return (
        classData.students.length > 25 || // Class size violation
        metrics.bullyingCount > 2 || // Bullying count violation
        metrics.avgAcademic < 5 || // Low academic performance
        metrics.avgWellbeing < 5 // Low wellbeing
    );
}

// Update violation flags
function updateViolationFlags(classData) {
    const flagsContainer = document.getElementById('violationFlags');
    flagsContainer.innerHTML = '';

    const violations = [];
    const metrics = calculateClassMetrics(classData);

    if (classData.students.length > 25) {
        violations.push('Class size exceeds limit (25)');
    }
    if (metrics.bullyingCount > 2) {
        violations.push('High bullying count');
    }
    if (metrics.avgAcademic < 5) {
        violations.push('Low academic performance');
    }
    if (metrics.avgWellbeing < 5) {
        violations.push('Low wellbeing score');
    }

    violations.forEach(violation => {
        const flag = document.createElement('div');
        flag.className = 'violation-flag';
        flag.innerHTML = `<span class="icon">⚠️</span> ${violation}`;
        flagsContainer.appendChild(flag);
    });
}

// Update student table
function updateStudentTable(students) {
    const tbody = document.getElementById('studentsTableBody');
    tbody.innerHTML = '';

    const filteredStudents = filterStudents(students);
    const sortedStudents = sortStudents(filteredStudents);

    sortedStudents.forEach(student => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${student.id}</td>
            <td>${student.academicScore.toFixed(1)}</td>
            <td>${student.wellbeingScore.toFixed(1)}</td>
            <td>${student.bullyingScore.toFixed(1)}</td>
            <td>${generateStudentFlags(student)}</td>
            <td>
                <button class="move-btn" onclick="showReassignModal(${student.id})">
                    <span class="icon">🔄</span> Move
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Generate student flags
function generateStudentFlags(student) {
    const flags = [];

    if (student.academicScore < 5) {
        flags.push('<span class="student-flag flag-needs-attention">📚 Low Academic</span>');
    }
    if (student.wellbeingScore < 5) {
        flags.push('<span class="student-flag flag-high-risk">⚠️ Low Wellbeing</span>');
    }
    if (student.bullyingScore > 7) {
        flags.push('<span class="student-flag flag-bullying">⚠️ Bullying Risk</span>');
    }

    return flags.join(' ');
}

// Filter students based on current filters
function filterStudents(students) {
    return students.filter(student => {
        // Apply range filters
        if (currentFilters.academic.min !== null && student.academicScore < currentFilters.academic.min) return false;
        if (currentFilters.academic.max !== null && student.academicScore > currentFilters.academic.max) return false;
        if (currentFilters.wellbeing.min !== null && student.wellbeingScore < currentFilters.wellbeing.min) return false;
        if (currentFilters.wellbeing.max !== null && student.wellbeingScore > currentFilters.wellbeing.max) return false;

        // Apply flag filters
        if (currentFilters.flags.highRisk && student.wellbeingScore >= 5) return false;
        if (currentFilters.flags.needsAttention && student.academicScore >= 5) return false;
        if (currentFilters.flags.bullying && student.bullyingScore <= 7) return false;

        return true;
    });
}

// Sort students
function sortStudents(students) {
    if (!sortConfig.column) return students;

    return [...students].sort((a, b) => {
        let comparison = 0;
        switch (sortConfig.column) {
            case 'id':
                comparison = a.id - b.id;
                break;
            case 'academic':
                comparison = a.academicScore - b.academicScore;
                break;
            case 'wellbeing':
                comparison = a.wellbeingScore - b.wellbeingScore;
                break;
            case 'bullying':
                comparison = a.bullyingScore - b.bullyingScore;
                break;
        }
        return sortConfig.direction === 'asc' ? comparison : -comparison;
    });
}

// Sort table
function sortTable(column) {
    if (sortConfig.column === column) {
        sortConfig.direction = sortConfig.direction === 'asc' ? 'desc' : 'asc';
    } else {
        sortConfig.column = column;
        sortConfig.direction = 'asc';
    }

    // Update sort indicators
    document.querySelectorAll('th[data-sort]').forEach(th => {
        const icon = th.querySelector('.sort-icon');
        if (th.dataset.sort === column) {
            icon.textContent = sortConfig.direction === 'asc' ? '↑' : '↓';
        } else {
            icon.textContent = '↕️';
        }
    });

    // Refresh table
    if (currentClassId !== null) {
        updateStudentTable(allocationData.classes[currentClassId].students);
    }
}

// Filter class list
function filterClassList(event) {
    const searchTerm = event.target.value.toLowerCase();
    document.querySelectorAll('.class-item').forEach(item => {
        const className = item.querySelector('.class-name').textContent.toLowerCase();
        item.style.display = className.includes(searchTerm) ? 'flex' : 'none';
    });
}

// Filter student table
function filterStudentTable(event) {
    const searchTerm = event.target.value.toLowerCase();
    document.querySelectorAll('#studentsTableBody tr').forEach(row => {
        const studentId = row.querySelector('td').textContent.toLowerCase();
        row.style.display = studentId.includes(searchTerm) ? '' : 'none';
    });
}

// Show filter modal
function showFilterModal() {
    const modal = document.getElementById('filterModal');
    modal.classList.add('active');
}

// Close filter modal
function closeFilterModal() {
    const modal = document.getElementById('filterModal');
    modal.classList.remove('active');
}

// Apply filters
function applyFilters() {
    // Update filter state
    currentFilters = {
        academic: {
            min: parseFloat(document.getElementById('academicMin').value) || null,
            max: parseFloat(document.getElementById('academicMax').value) || null
        },
        wellbeing: {
            min: parseFloat(document.getElementById('wellbeingMin').value) || null,
            max: parseFloat(document.getElementById('wellbeingMax').value) || null
        },
        flags: {
            highRisk: document.getElementById('filterHighRisk').checked,
            needsAttention: document.getElementById('filterNeedsAttention').checked,
            bullying: document.getElementById('filterBullying').checked
        }
    };

    // Refresh table
    if (currentClassId !== null) {
        updateStudentTable(allocationData.classes[currentClassId].students);
    }

    closeFilterModal();
}

// Show reassign modal
function showReassignModal(studentId) {
    const modal = document.getElementById('reassignModal');
    document.getElementById('moveStudentId').textContent = studentId;

    // Populate destination class dropdown
    const select = document.getElementById('destinationClass');
    select.innerHTML = '';
    allocationData.classes.forEach((_, index) => {
        if (index !== currentClassId) {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Class ${index}`;
            select.appendChild(option);
        }
    });

    modal.classList.add('active');
}

// Close reassign modal
function closeReassignModal() {
    const modal = document.getElementById('reassignModal');
    modal.classList.remove('active');
}

// Confirm student move
function confirmMove() {
    const studentId = parseInt(document.getElementById('moveStudentId').textContent);
    const destinationClassId = parseInt(document.getElementById('destinationClass').value);

    // Find student in current class
    const sourceClass = allocationData.classes[currentClassId];
    const studentIndex = sourceClass.students.findIndex(s => s.id === studentId);
    const student = sourceClass.students[studentIndex];

    // Remove from current class
    sourceClass.students.splice(studentIndex, 1);

    // Add to destination class
    allocationData.classes[destinationClassId].students.push(student);

    // Save changes to localStorage
    localStorage.setItem('allocationResults', JSON.stringify(allocationData));

    // Refresh UI
    populateClassList();
    loadClassDetails(currentClassId);

    closeReassignModal();
}

// Show error message
function showError(message) {
    const container = document.querySelector('.classroom-container');
    container.innerHTML = `
        <div class="error-message">
            <span class="icon">⚠️</span>
            <h4>Error</h4>
            <p>${message}</p>
            <button class="action-btn" onclick="window.location.href='allocation.html'">
                <span class="icon">↩</span> Go to Allocation
            </button>
        </div>
    `;
} 