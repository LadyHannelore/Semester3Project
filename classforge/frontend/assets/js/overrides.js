// Performance constants
const DEBOUNCE_DELAY = 300; // ms for input changes
const THROTTLE_DELAY = 16; // ms for 60fps animations
const BATCH_SIZE = 50; // Number of DOM operations per batch
const ANIMATION_DURATION = 150; // ms for drag animations

// Error tracking
const ERROR_LOG_ENDPOINT = '/api/log-error';
const MAX_ERROR_RETRY = 3;

// Performance monitoring
let renderTimes = [];
let interactionTimes = [];

// Utility functions for performance optimization
function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            try {
                func.apply(this, args);
            } catch (error) {
                handleError(error, 'debounce');
            }
        }, delay);
    };
}

function throttle(func, delay) {
    let lastCall = 0;
    return function (...args) {
        const now = new Date().getTime();
        if (now - lastCall < delay) return;
        lastCall = now;
        try {
            func.apply(this, args);
        } catch (error) {
            handleError(error, 'throttle');
        }
    };
}

// Optimized batch DOM operations
function batchDOMUpdates(operations) {
    return new Promise(resolve => {
        requestAnimationFrame(() => {
            operations.forEach(op => op());
            resolve();
        });
    });
}

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

// Initialize event listeners with optimized handlers
function initializeEventListeners() {
    // View toggle button with throttling
    const viewToggleBtn = document.getElementById('viewToggleBtn');
    if (viewToggleBtn) {
        viewToggleBtn.addEventListener('click', throttle(toggleView, THROTTLE_DELAY));
    }

    // Action buttons with debouncing for save operations
    const saveBtn = document.getElementById('saveBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', debounce(saveChanges, DEBOUNCE_DELAY));
    }
    
    const undoBtn = document.getElementById('undoBtn');
    if (undoBtn) {
        undoBtn.addEventListener('click', undoLastMove);
    }
    
    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetToAutoAllocation);
    }

    // Add student form with optimized submission
    const addStudentForm = document.getElementById('addStudentForm');
    if (addStudentForm) {
        addStudentForm.addEventListener('submit', debounce(handleAddStudentFormSubmit, DEBOUNCE_DELAY));
    }
}

// Initialize class grid with Sortable.js and optimized rendering
async function initializeClassGrid() {
    const classGrid = document.getElementById('classGrid');
    classGrid.innerHTML = '';

    // Batch DOM operations for better performance
    const operations = allocationData.classes.map((classData, index) => 
        () => {
            const column = createClassColumn(index, classData);
            classGrid.appendChild(column);

            // Initialize Sortable.js for drag and drop with optimized settings
            new Sortable(column.querySelector('.class-content'), {
                group: 'classes',
                animation: ANIMATION_DURATION,
                ghostClass: 'sortable-ghost',
                chosenClass: 'sortable-chosen',
                forceFallback: false, // Use native HTML5 DnD when possible
                fallbackTolerance: 3,
                onEnd: throttle(handleStudentMove, THROTTLE_DELAY)
            });
        }
    );

    // Execute operations in batches to prevent UI blocking
    for (let i = 0; i < operations.length; i += BATCH_SIZE) {
        const batch = operations.slice(i, i + BATCH_SIZE);
        await batchDOMUpdates(batch);
    }
}

// Create class column
function createClassColumn(index, classData) {
    const column = document.createElement('div');
    column.className = 'class-column';
    column.dataset.classId = index; // Add classId here
    column.innerHTML = `
        <div class="class-header">
            <h4><span class="icon">🏫</span> Class ${index}</h4>
            <span class="class-size">${classData.students.length} students</span>
        </div>
        <div class="class-content">
            <!-- Student cards will be appended here -->
        </div>
    `;
    const classContent = column.querySelector('.class-content');
    if (classData.students && classContent) {
        classData.students.forEach(student => {
            const studentCardElement = createStudentCard(student);
            if (studentCardElement) { // Ensure card was created
                classContent.appendChild(studentCardElement);
            }
        });
    }
    return column;
}

// Create student card from template and set data
function createStudentCard(student) {
    const template = document.getElementById('studentCardTemplate');
    if (!template) {
        console.error("Student card template not found.");
        return null;
    }
    const cardFragment = template.content.cloneNode(true);
    const cardElement = cardFragment.firstElementChild;

    if (!cardElement) {
        console.error("Student card template is empty or structured incorrectly for student:", student);
        const div = document.createElement('div');
        div.className = 'student-card';
        div.textContent = `Error: Card for ${student.id}`;
        div.dataset.studentId = student.id;
        return div;
    }

    cardElement.dataset.studentId = student.id;

    // Set student data
    const studentIdDisplay = cardElement.querySelector('.student-id');
    if (studentIdDisplay) studentIdDisplay.textContent = `Student ${student.id}`;

    const academicDisplay = cardElement.querySelector('.academic');
    if (academicDisplay) academicDisplay.textContent = student.academicScore !== undefined ? student.academicScore.toFixed(1) : 'N/A';

    const wellbeingDisplay = cardElement.querySelector('.wellbeing');
    if (wellbeingDisplay) wellbeingDisplay.textContent = student.wellbeingScore !== undefined ? student.wellbeingScore.toFixed(1) : 'N/A';

    const bullyingDisplay = cardElement.querySelector('.bullying');
    if (bullyingDisplay) bullyingDisplay.textContent = student.bullyingScore !== undefined ? student.bullyingScore.toFixed(1) : 'N/A';

    // Add status indicators
    const indicators = cardElement.querySelector('.status-indicators');
    if (indicators) {
        if (student.academicScore < 5) indicators.appendChild(createStatusIndicator('red'));
        if (student.wellbeingScore < 5) indicators.appendChild(createStatusIndicator('yellow'));
        if (student.bullyingScore > 7) indicators.appendChild(createStatusIndicator('red'));
    }
    return cardElement;
}

// Create status indicator
function createStatusIndicator(type) {
    const indicator = document.createElement('div');
    indicator.className = `status-indicator status-${type}`;
    return indicator;
}

// Handle student move
function handleStudentMove(evt) {
    const fromClassId = parseInt(evt.from.closest('.class-column').dataset.classId);
    const toClassId = parseInt(evt.to.closest('.class-column').dataset.classId);
    const studentId = evt.item.dataset.studentId; // Get ID from data attribute

    // Record move in history
    moveHistory.push({
        studentId,
        fromClass: fromClassId,
        toClass: toClassId
    });

    // Update data
    const student = allocationData.classes[fromClassId].students.find(s => String(s.id) === String(studentId));
    
    if (!student) {
        console.error(`Student with ID ${studentId} not found in class ${fromClassId}. UI might be inconsistent.`);
        // Re-render to attempt to fix UI discrepancy, or show an error to the user.
        showError(`Error moving student ${studentId}. Data inconsistency detected. Please refresh or reset.`);
        initializeClassGrid(); // Re-render to reflect actual data state
        updateValidation();
        return;
    }

    allocationData.classes[fromClassId].students = allocationData.classes[fromClassId].students.filter(s => String(s.id) !== String(studentId));
    allocationData.classes[toClassId].students.push(student);

    // Update UI
    updateClassHeaders();
    updateValidation();
}

// Generate a new unique student ID
function generateNewStudentId() {
    let maxIdNum = 0;
    if (allocationData && allocationData.classes) {
        allocationData.classes.forEach(cls => {
            cls.students.forEach(student => {
                if (student.id && typeof student.id === 'string' && student.id.startsWith('S')) {
                    const numPart = parseInt(student.id.substring(1));
                    if (!isNaN(numPart) && numPart > maxIdNum) {
                        maxIdNum = numPart;
                    }
                }
            });
        });
    }
    return `S${maxIdNum + 1}`;
}

// Handle Add Student Form Submission
function handleAddStudentFormSubmit(event) {
    event.preventDefault();
    if (!allocationData || !allocationData.classes) {
        showError("Allocation data is not loaded. Cannot add student.");
        return;
    }

    const studentId = generateNewStudentId(); // Auto-generate ID
    const academicScore = parseFloat(document.getElementById('newAcademicScore').value);
    const wellbeingScore = parseFloat(document.getElementById('newWellbeingScore').value);
    const bullyingScore = parseFloat(document.getElementById('newBullyingScore').value);

    // Student ID is auto-generated, so no need to check if it's empty.
    // The generateNewStudentId function aims to create a unique ID.
    // The studentExists check below will serve as a final safeguard.

    if (isNaN(academicScore) || academicScore < 0 || academicScore > 100) {
        showError("Invalid Academic Score. Must be between 0 and 100.");
        return;
    }
    if (isNaN(wellbeingScore) || wellbeingScore < 0 || wellbeingScore > 10) {
        showError("Invalid Wellbeing Score. Must be between 0 and 10.");
        return;
    }
    if (isNaN(bullyingScore) || bullyingScore < 0 || bullyingScore > 10) {
        showError("Invalid Bullying Score. Must be between 0 and 10.");
        return;
    }

    // Check if student ID already exists
    const studentExists = allocationData.classes.some(cls => cls.students.some(s => String(s.id) === String(studentId)));
    if (studentExists) {
        showError(`Generated Student ID ${studentId} somehow already exists. Please try again or check data.`);
        return;
    }

    const newStudent = {
        id: studentId, // Can be string or number, ensure consistency
        academicScore: academicScore,
        wellbeingScore: wellbeingScore,
        bullyingScore: bullyingScore
        // Add other relevant properties if your student objects have them
    };

    const bestClassIndex = findBestClassForNewStudent(newStudent);

    if (bestClassIndex !== -1) {
        allocationData.classes[bestClassIndex].students.push(newStudent);
        showSuccess(`Student ${studentId} added to Class ${bestClassIndex}.`);
        document.getElementById('addStudentForm').reset(); // Clear the form
        
        // Refresh UI
        initializeClassGrid(); // Easiest way to update the grid with the new card
        updateClassHeaders();
        updateValidation();
    } else {
        showError("No suitable class found. All classes might be full or adding the student would violate constraints (e.g., bully limit).");
    }
}

// Find the best class for a new student
function findBestClassForNewStudent(student) {
    let bestClassIndex = -1;
    let minSuitableClassSize = Infinity;

    if (!allocationData || !allocationData.classes) return -1;

    allocationData.classes.forEach((classData, index) => {
        const currentClassSize = classData.students.length;
        
        if (currentClassSize >= MAX_CLASS_SIZE) {
            return; // Class is full
        }

        if (student.bullyingScore > 7) { // Assuming > 7 is a "bully"
            const bullyCount = classData.students.filter(s => s.bullyingScore > 7).length;
            if (bullyCount >= MAX_BULLIES_PER_CLASS) {
                return; // Class has reached bully limit
            }
        }

        // This class is suitable, check if it's the smallest suitable one found so far
        if (currentClassSize < minSuitableClassSize) {
            minSuitableClassSize = currentClassSize;
            bestClassIndex = index;
        }
    });

    return bestClassIndex;
}

// Update class headers
function updateClassHeaders() {
    document.querySelectorAll('.class-column').forEach((column) => {
        const classId = parseInt(column.dataset.classId);
        if (allocationData && allocationData.classes && allocationData.classes[classId]) {
            const size = allocationData.classes[classId].students.length;
            column.querySelector('.class-size').textContent = `${size} students`;
            column.classList.toggle('over-limit', size > MAX_CLASS_SIZE);
        }
    });
}

// Update validation feedback
function updateValidation() {
    if (!allocationData || !allocationData.classes) return;

    // Class size validation
    const classSizeValidation = document.getElementById('classSizeValidation');
    let overLimitClasses = [];
    allocationData.classes.forEach((c, index) => {
        if (c.students.length > MAX_CLASS_SIZE) {
            overLimitClasses.push(index);
        }
    });
    if (overLimitClasses.length > 0) {
        classSizeValidation.textContent = `⚠️ Classes ${overLimitClasses.join(', ')} exceed limit (${MAX_CLASS_SIZE} students).`;
        classSizeValidation.className = 'validation-content warning';
    } else {
        classSizeValidation.textContent = '✅ All classes within limit.';
        classSizeValidation.className = 'validation-content';
    }

    // Bullying validation
    const bullyingValidation = document.getElementById('bullyingValidation');
    let highBullyingClasses = [];
    allocationData.classes.forEach((c, index) => {
        const bullyCount = c.students.filter(s => s.bullyingScore > 7).length;
        if (bullyCount > MAX_BULLIES_PER_CLASS) {
            highBullyingClasses.push(index);
        }
    });
    if (highBullyingClasses.length > 0) {
        bullyingValidation.textContent = `⚠️ High bullying concentration in Classes ${highBullyingClasses.join(', ')} (max ${MAX_BULLIES_PER_CLASS} bullies).`;
        bullyingValidation.className = 'validation-content warning';
    } else {
        bullyingValidation.textContent = '✅ Bullying well distributed.';
        bullyingValidation.className = 'validation-content';
    }

    // Academic balance validation
    const academicValidation = document.getElementById('academicValidation');
    let academicImbalanceClasses = [];
    allocationData.classes.forEach((c, index) => {
        const stdDev = calculateStdDev(c.students.map(s => s.academicScore));
        if (stdDev > 2) { // Assuming threshold of 2 for std dev
            academicImbalanceClasses.push(index);
        }
    });
    if (academicImbalanceClasses.length > 0) {
        academicValidation.textContent = `⚠️ Academic imbalance in Classes ${academicImbalanceClasses.join(', ')}.`;
        academicValidation.className = 'validation-content warning';
    } else {
        academicValidation.textContent = '✅ Academic balance good.';
        academicValidation.className = 'validation-content';
    }

    // Wellbeing balance validation
    const wellbeingValidation = document.getElementById('wellbeingValidation');
    let wellbeingImbalanceClasses = [];
    allocationData.classes.forEach((c, index) => {
        const stdDev = calculateStdDev(c.students.map(s => s.wellbeingScore));
        if (stdDev > 2) { // Assuming threshold of 2 for std dev
            wellbeingImbalanceClasses.push(index);
        }
    });
    if (wellbeingImbalanceClasses.length > 0) {
        wellbeingValidation.textContent = `⚠️ Wellbeing imbalance in Classes ${wellbeingImbalanceClasses.join(', ')}.`;
        wellbeingValidation.className = 'validation-content warning';
    } else {
        wellbeingValidation.textContent = '✅ Wellbeing balance good.';
        wellbeingValidation.className = 'validation-content';
    }
}

// Calculate standard deviation
function calculateStdDev(values) {
    if (!values || values.length === 0) {
        return 0; // Or NaN, depending on how you want to handle empty/invalid input
    }
    const mean = values.reduce((sum, val) => sum + (val || 0), 0) / values.length; // Handle potential null/undefined in scores
    const squareDiffs = values.map(val => Math.pow((val || 0) - mean, 2));
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

// Error handling
function handleError(error, context = 'unknown') {
    console.error(`[${context}] Error:`, error);
    
    // Display user-friendly message
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.textContent = 'Something went wrong. Please try again.';
    errorElement.style.cssText = 'position: fixed; bottom: 20px; right: 20px; background-color: #f8d7da; color: #721c24; padding: 10px 20px; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); z-index: 1000;';
    document.body.appendChild(errorElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (document.body.contains(errorElement)) {
            document.body.removeChild(errorElement);
        }
    }, 5000);
    
    // Log to server if possible
    try {
        logErrorToServer(error, context);
    } catch (loggingError) {
        console.error('Failed to log error:', loggingError);
    }
}

// Error logging
function logErrorToServer(error, context) {
    const errorData = {
        message: error.message || 'Unknown error',
        stack: error.stack || '',
        context: context,
        timestamp: new Date().toISOString(),
        url: window.location.href,
        userAgent: navigator.userAgent
    };
    
    // Use sendBeacon if available for non-blocking logging
    if (navigator.sendBeacon) {
        const blob = new Blob([JSON.stringify(errorData)], { type: 'application/json' });
        navigator.sendBeacon(ERROR_LOG_ENDPOINT, blob);
        return;
    }
    
    // Fallback to fetch with keepalive
    fetch(ERROR_LOG_ENDPOINT, {
        method: 'POST',
        body: JSON.stringify(errorData),
        headers: {
            'Content-Type': 'application/json'
        },
        keepalive: true
    }).catch(e => console.error('Error logging failed:', e));
}

// Performance monitoring
function trackRender(componentName, startTime) {
    const endTime = performance.now();
    const renderTime = endTime - startTime;
    renderTimes.push({ component: componentName, time: renderTime });
    
    if (renderTime > 100) { // Slow render threshold
        console.warn(`Slow render detected in ${componentName}: ${renderTime.toFixed(2)}ms`);
    }
}

function trackInteraction(interactionName, startTime) {
    const endTime = performance.now();
    interactionTimes.push({ 
        interaction: interactionName, 
        time: endTime - startTime,
        timestamp: Date.now()
    });
}