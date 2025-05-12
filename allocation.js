// Global state
let selectedStrategy = null;
let allocationInProgress = false;
let allocationResults = null;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    initializeStrategySelection();
    initializeParameterControls();
    initializeRunButton();
    initializeActionButtons();
});

// Strategy selection
function initializeStrategySelection() {
    // Only the genetic algorithm strategy is available
    const geneticRadio = document.getElementById('genetic');
    if (geneticRadio) {
        geneticRadio.checked = true;
        selectedStrategy = 'genetic';
    } else {
        selectedStrategy = null; // Should not happen if HTML is correct
    }
    updateParameterVisibility();
    updateRunButtonState();
}

// Parameter controls
function initializeParameterControls() {
    // Genetic algorithm parameters
    const geneticInputs = document.querySelectorAll('#genetic-params input');
    geneticInputs.forEach(input => {
        input.addEventListener('change', validateNumberInput);
        if (input.type === 'range') {
            const valueSpan = input.nextElementSibling;
            if (valueSpan && valueSpan.classList.contains('slider-value')) {
                // Add event listener to update slider value display
                input.addEventListener('input', (event) => {
                    valueSpan.textContent = event.target.id === 'mutationRate' ? `${event.target.value}%` : event.target.value;
                });
                // Set initial display value
                valueSpan.textContent = input.id === 'mutationRate' ? `${input.value}%` : input.value;
            }
        }
    });
}

// Update parameter visibility based on selected strategy
function updateParameterVisibility() {
    // Only genetic-params should be visible
    const geneticParamsGroup = document.getElementById('genetic-params');
    if (geneticParamsGroup) {
        geneticParamsGroup.style.display = 'grid';
    }
    // Hide other parameter groups if they existed
    document.querySelectorAll('.parameter-group').forEach(group => {
        if (group.id !== 'genetic-params') {
            group.style.display = 'none';
        }
    });
}

// Validate number input
function validateNumberInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);

    if (isNaN(value) || value < min || value > max) {
        input.value = Math.min(Math.max(min, value), max);
    }
}

// Run button functionality
function initializeRunButton() {
    const runButton = document.querySelector('.run-button');
    if (runButton) {
        runButton.addEventListener('click', startAllocation);
    }
}

function updateRunButtonState() {
    const runButton = document.querySelector('.run-button');
    if (runButton) {
        runButton.disabled = !selectedStrategy || allocationInProgress;
    }
}

async function startAllocation() {
    if (allocationInProgress) return;

    allocationInProgress = true;
    updateRunButtonState();
    showProgressIndicator();

    try {
        // Collect parameters for the genetic algorithm
        const gaParams = {
            populationSize: parseInt(document.getElementById('populationSize').value),
            generations: parseInt(document.getElementById('generations').value),
            mutationRate: parseFloat(document.getElementById('mutationRate').value) / 100, // Convert percentage to decimal
            maxClassSize: parseInt(document.getElementById('maxClassSize').value),
            academicBalanceWeight: parseFloat(document.getElementById('academicBalanceWeight').value),
            wellbeingBalanceWeight: parseFloat(document.getElementById('wellbeingBalanceWeight').value),
            maxBulliesPerClass: parseInt(document.getElementById('maxBulliesPerClass').value),
            maxFriendsSplit: parseInt(document.getElementById('maxFriendsSplit').value)
        };
        
        await simulateGeneticAlgorithmAllocation(gaParams);
        showResults();
    } catch (error) {
        showError(error.message);
    } finally {
        allocationInProgress = false;
        updateRunButtonState();
    }
}

function showProgressIndicator() {
    const progressContainer = document.querySelector('.progress-indicator');
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Initializing allocation...';
}

// Helper to get currentDataset from localStorage
function getStoredDataset() {
    const stored = localStorage.getItem('classforgeDataset');
    if (stored) {
        return JSON.parse(stored);
    }
    return null;
}

// Simulate Genetic Algorithm Allocation
async function simulateGeneticAlgorithmAllocation(params) {
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    progressFill.style.width = `10%`;
    progressText.textContent = 'Preparing data...';

    const dataset = getStoredDataset();
    if (!dataset || !dataset.rows || dataset.rows.length === 0) {
        throw new Error("No student data found. Please upload or generate data first.");
    }

    // Prepare students array (map to Python expected fields)
    const students = dataset.rows.map(row => {
        const studentIdStr = row[dataset.headers.indexOf('Student_ID')];
        let studentId = studentIdStr;
        return {
            id: studentId,
            academicScore: parseFloat(row[dataset.headers.indexOf('Academic_Performance')]),
            wellbeingScore: parseFloat(row[dataset.headers.indexOf('Wellbeing_Score')]),
            bullyingScore: parseFloat(row[dataset.headers.indexOf('Bullying_Score')])
        };
    });

    progressFill.style.width = `30%`;
    progressText.textContent = 'Sending data to optimizer...';

    // Call Python backend
    let response;
    try {
        response = await fetch('http://127.0.0.1:5001/allocate', { // Ensure the URL matches the Flask server
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                students,
                params: {
                    maxClassSize: params.maxClassSize,
                    maxBulliesPerClass: params.maxBulliesPerClass,
                    generations: params.generations,
                    populationSize: params.populationSize
                }
            })
        });
    } catch (err) {
        throw new Error("Could not connect to backend optimizer. Is the Python server running?");
    }

    progressFill.style.width = `60%`;
    progressText.textContent = 'Processing results...';

    if (!response.ok) {
        const errText = await response.text();
        throw new Error("Backend error: " + errText);
    }
    const result = await response.json();

    if (!result.success) {
        throw new Error(result.error || "Allocation failed.");
    }

    // Save results in the same format as before
    allocationResults = {
        success: true,
        metrics: result.metrics,
        violations: result.violations,
        classes: result.classes
    };

    localStorage.setItem('allocationResults', JSON.stringify(allocationResults));
    localStorage.setItem('autoAllocationResults', JSON.stringify(allocationResults));

    progressFill.style.width = `100%`;
    progressText.textContent = 'Allocation complete.';
}

function showResults() {
    const resultsContainer = document.querySelector('.results-summary');
    const progressContainer = document.querySelector('.progress-indicator');

    progressContainer.style.display = 'none';
    resultsContainer.style.display = 'block';

    // Update metrics
    const metrics = allocationResults.metrics;
    document.getElementById('resultsTotalStudents').textContent = metrics.totalStudents || '-';
    document.getElementById('resultsNumClasses').textContent = metrics.numClasses || '-';
    document.getElementById('resultsAvgAcademic').textContent = metrics.avgAcademic ? metrics.avgAcademic.toFixed(1) : '-';
    document.getElementById('resultsAvgWellbeing').textContent = metrics.avgWellbeing ? metrics.avgWellbeing.toFixed(1) : '-';

    document.getElementById('balance-score').textContent = metrics.balanceScore ? (metrics.balanceScore * 100).toFixed(1) + '%' : '-';
    document.getElementById('diversity-score').textContent = metrics.diversityScore ? (metrics.diversityScore * 100).toFixed(1) + '%' : '-';
    document.getElementById('constraint-satisfaction').textContent = metrics.constraintSatisfaction ? (metrics.constraintSatisfaction * 100).toFixed(1) + '%' : '-';
    document.getElementById('processing-time').textContent = metrics.processingTime || '-';

    // Update violations
    const violationsList = document.querySelector('.constraint-violations ul');
    violationsList.innerHTML = '';
    allocationResults.violations.forEach(violation => {
        const li = document.createElement('li');
        li.textContent = violation;
        violationsList.appendChild(li);
    });

    // Populate results table
    const resultsTableBody = document.getElementById('resultsTableBody');
    resultsTableBody.innerHTML = '';
    allocationResults.classes.forEach((classData, index) => {
        // Compute avgAcademic and avgWellbeing for this class
        let avgAcademic = '-';
        let avgWellbeing = '-';
        if (classData.students && classData.students.length > 0) {
            const academicSum = classData.students.reduce((sum, s) => sum + (typeof s.academicScore === 'number' ? s.academicScore : parseFloat(s.academicScore) || 0), 0);
            const wellbeingSum = classData.students.reduce((sum, s) => sum + (typeof s.wellbeingScore === 'number' ? s.wellbeingScore : parseFloat(s.wellbeingScore) || 0), 0);
            avgAcademic = (academicSum / classData.students.length).toFixed(1);
            avgWellbeing = (wellbeingSum / classData.students.length).toFixed(1);
        }
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>Class ${index + 1}</td>
            <td>${classData.students.length}</td>
            <td>${avgAcademic}</td>
            <td>${avgWellbeing}</td>
        `;
        resultsTableBody.appendChild(row);
    });

    // Show success message
    const successMessage = document.querySelector('.success-message');
    successMessage.style.display = 'flex';

    // Enable view classroom button
    const viewClassroomBtn = document.getElementById('viewClassroomBtn');
    if (viewClassroomBtn) {
        viewClassroomBtn.disabled = false;
    }
}

function showError(message) {
    const progressContainer = document.querySelector('.progress-indicator');
    progressContainer.style.display = 'none';

    // Show error message
    const errorMessage = document.createElement('div');
    errorMessage.className = 'error-message';
    errorMessage.innerHTML = `
        <span class="icon">⚠️</span>
        <h4>Allocation Failed</h4>
        <p>${message}</p>
    `;
    document.querySelector('.run-container').appendChild(errorMessage);
}

// Action buttons
function initializeActionButtons() {
    const actionButtons = document.querySelectorAll('.action-btn');
    actionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const action = this.dataset.action;
            handleAction(action);
        });
    });
}

function handleAction(action) {
    switch (action) {
        case 'back':
            window.location.href = 'index.html';
            break;
        case 'view-classrooms':
            if (allocationResults && allocationResults.success) {
                window.location.href = 'classroom.html';
            } else {
                showError("Allocation not yet run or failed. Cannot view classrooms.");
            }
            break;
    }
}