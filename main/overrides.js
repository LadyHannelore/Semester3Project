// Global state
let allocationData = null;
let moveHistory = [];
const MAX_CLASS_SIZE = 25;
const MAX_BULLIES_PER_CLASS = 2;

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
        initializeClassGrid();
        updateValidation();
    } else {
        showError('No allocation data found. Please run allocation first.');
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // View toggle button
    document.getElementById('viewToggleBtn').addEventListener('click', toggleView);

    // Action buttons
    document.getElementById('saveBtn').addEventListener('click', saveChanges);
    document.getElementById('undoBtn').addEventListener('click', undoLastMove);
    document.getElementById('resetBtn').addEventListener('click', resetToAutoAllocation);
}

// Initialize class grid with Sortable.js
function initializeClassGrid() {
    const classGrid = document.getElementById('classGrid');
    classGrid.innerHTML = '';

    allocationData.classes.forEach((classData, index) => {
        const column = createClassColumn(index, classData);
        classGrid.appendChild(column);

        // Initialize Sortable.js for drag and drop
        new Sortable(column.querySelector('.class-content'), {
            group: 'classes',
            animation: 150,
            ghostClass: 'sortable-ghost',
            chosenClass: 'sortable-chosen',
            onEnd: function(evt) {
                handleStudentMove(evt);
            }
        });
    });
}

// Create class column
function createClassColumn(index, classData) {
    const column = document.createElement('div');
    column.className = 'class-column';
    column.innerHTML = `
        <div class="class-header">
            <h4><span class="icon">🏫</span> Class ${index}</h4>
            <span class="class-size">${classData.students.length} students</span>
        </div>
        <div class="class-content">
            ${classData.students.map(student => createStudentCard(student)).join('')}
        </div>
    `;
    return column;
}

// Create student card
function createStudentCard(student) {
    const template = document.getElementById('studentCardTemplate');
    const card = template.content.cloneNode(true);
    
    // Set student data
    card.querySelector('.student-id').textContent = `Student ${student.id}`;
    card.querySelector('.academic').textContent = student.academicScore.toFixed(1);
    card.querySelector('.wellbeing').textContent = student.wellbeingScore.toFixed(1);
    card.querySelector('.bullying').textContent = student.bullyingScore.toFixed(1);

    // Add status indicators
    const indicators = card.querySelector('.status-indicators');
    if (student.academicScore < 5) {
        indicators.appendChild(createStatusIndicator('red'));
    }
    if (student.wellbeingScore < 5) {
        indicators.appendChild(createStatusIndicator('yellow'));
    }
    if (student.bullyingScore > 7) {
        indicators.appendChild(createStatusIndicator('red'));
    }

    return card;
}

// Create status indicator
function createStatusIndicator(type) {
    const indicator = document.createElement('div');
    indicator.className = `status-indicator status-${type}`;
    return indicator;
}

// Handle student move
function handleStudentMove(evt) {
    const fromClass = parseInt(evt.from.closest('.class-column').dataset.classId);
    const toClass = parseInt(evt.to.closest('.class-column').dataset.classId);
    const studentId = parseInt(evt.item.querySelector('.student-id').textContent.split(' ')[1]);

    // Record move in history
    moveHistory.push({
        studentId,
        fromClass,
        toClass
    });

    // Update data
    const student = allocationData.classes[fromClass].students.find(s => s.id === studentId);
    allocationData.classes[fromClass].students = allocationData.classes[fromClass].students.filter(s => s.id !== studentId);
    allocationData.classes[toClass].students.push(student);

    // Update UI
    updateClassHeaders();
    updateValidation();
}

// Update class headers
function updateClassHeaders() {
    document.querySelectorAll('.class-column').forEach((column, index) => {
        const size = allocationData.classes[index].students.length;
        column.querySelector('.class-size').textContent = `${size} students`;
        
        // Add warning class if over limit
        column.classList.toggle('over-limit', size > MAX_CLASS_SIZE);
    });
}

// Update validation feedback
function updateValidation() {
    // Class size validation
    const classSizes = allocationData.classes.map(c => c.students.length);
    const sizeValidation = document.getElementById('classSizeValidation');
    const overLimit = classSizes.some(size => size > MAX_CLASS_SIZE);
    sizeValidation.textContent = overLimit ? '⚠️ Some classes exceed limit' : '✅ All classes within limit';
    sizeValidation.className = `validation-content ${overLimit ? 'warning' : ''}`;

    // Bullying validation
    const bullyingCounts = allocationData.classes.map(c => 
        c.students.filter(s => s.bullyingScore > 7).length
    );
    const bullyingValidation = document.getElementById('bullyingValidation');
    const bullyingIssue = bullyingCounts.some(count => count > MAX_BULLIES_PER_CLASS);
    bullyingValidation.textContent = bullyingIssue ? '⚠️ High bullying concentration' : '✅ Bullying well distributed';
    bullyingValidation.className = `validation-content ${bullyingIssue ? 'warning' : ''}`;

    // Academic balance validation
    const academicStdDevs = allocationData.classes.map(c => 
        calculateStdDev(c.students.map(s => s.academicScore))
    );
    const academicValidation = document.getElementById('academicValidation');
    const academicIssue = academicStdDevs.some(stdDev => stdDev > 2);
    academicValidation.textContent = academicIssue ? '⚠️ Academic imbalance' : '✅ Academic balance good';
    academicValidation.className = `validation-content ${academicIssue ? 'warning' : ''}`;

    // Wellbeing balance validation
    const wellbeingStdDevs = allocationData.classes.map(c => 
        calculateStdDev(c.students.map(s => s.wellbeingScore))
    );
    const wellbeingValidation = document.getElementById('wellbeingValidation');
    const wellbeingIssue = wellbeingStdDevs.some(stdDev => stdDev > 2);
    wellbeingValidation.textContent = wellbeingIssue ? '⚠️ Wellbeing imbalance' : '✅ Wellbeing balance good';
    wellbeingValidation.className = `validation-content ${wellbeingIssue ? 'warning' : ''}`;
}

// Calculate standard deviation
function calculateStdDev(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squareDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((sum, val) => sum + val, 0) / squareDiffs.length;
    return Math.sqrt(avgSquareDiff);
}

// Toggle view (grid/list)
function toggleView() {
    const grid = document.querySelector('.class-grid');
    grid.classList.toggle('list-view');
}

// Save changes
function saveChanges() {
    localStorage.setItem('allocationResults', JSON.stringify(allocationData));
    showSuccess('Changes saved successfully');
}

// Undo last move
function undoLastMove() {
    if (moveHistory.length === 0) {
        showError('No moves to undo');
        return;
    }

    const lastMove = moveHistory.pop();
    const student = allocationData.classes[lastMove.toClass].students.find(s => s.id === lastMove.studentId);
    
    // Move student back
    allocationData.classes[lastMove.toClass].students = allocationData.classes[lastMove.toClass].students.filter(s => s.id !== lastMove.studentId);
    allocationData.classes[lastMove.fromClass].students.push(student);

    // Update UI
    initializeClassGrid();
    updateValidation();
}

// Reset to auto allocation
function resetToAutoAllocation() {
    if (confirm('Are you sure you want to reset all manual changes?')) {
        const stored = localStorage.getItem('autoAllocationResults');
        if (stored) {
            allocationData = JSON.parse(stored);
            moveHistory = [];
            initializeClassGrid();
            updateValidation();
            showSuccess('Reset to auto allocation');
        } else {
            showError('No auto allocation data found');
        }
    }
}

// Show error message
function showError(message) {
    // Implementation depends on your UI notification system
    alert(message);
}

// Show success message
function showSuccess(message) {
    // Implementation depends on your UI notification system
    alert(message);
} 