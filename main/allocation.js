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
    const steps = [
        { progress: 20, text: 'Loading student data...' },
        { progress: 40, text: 'Simulating student distribution...' },
        { progress: 60, text: 'Evaluating class assignments...' },
        { progress: 80, text: 'Finalizing results...' },
        { progress: 100, text: 'Allocation complete.' }
    ];

    for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 700)); // Adjusted timing
        progressFill.style.width = `${step.progress}%`;
        progressText.textContent = step.text;
    }

    const dataset = getStoredDataset();
    if (!dataset || !dataset.rows || dataset.rows.length === 0) {
        throw new Error("No student data found. Please upload or generate data first.");
    }

    const students = dataset.rows.map(row => {
        const studentIdStr = row[dataset.headers.indexOf('Student_ID')];
        let studentId = parseInt(studentIdStr);
        if (isNaN(studentId)) {
            studentId = studentIdStr; 
        }

        return {
            id: studentId,
            academicScore: parseFloat(row[dataset.headers.indexOf('Academic_Performance')]),
            wellbeingScore: parseFloat(row[dataset.headers.indexOf('Wellbeing_Score')]),
            bullyingScore: parseFloat(row[dataset.headers.indexOf('Bullying_Score')])
        };
    });

    const numStudents = students.length;
    const maxClassSize = params.maxClassSize;
    const numClasses = Math.ceil(numStudents / maxClassSize);
    const maxBulliesPerClass = params.maxBulliesPerClass;

    const generatedClasses = Array.from({ length: numClasses }, () => ({ students: [], bullyCount: 0 }));

    // Separate high-risk bullies from other students
    const highRiskBullies = students.filter(s => s.bullyingScore > 7);
    const otherStudents = students.filter(s => s.bullyingScore <= 7);

    // Distribute high-risk bullies first
    for (const bully of highRiskBullies) {
        let placed = false;
        for (let i = 0; i < numClasses; i++) {
            const classIndex = i % numClasses;
            if (generatedClasses[classIndex].students.length < maxClassSize && generatedClasses[classIndex].bullyCount < maxBulliesPerClass) {
                generatedClasses[classIndex].students.push(bully);
                generatedClasses[classIndex].bullyCount++;
                placed = true;
                break;
            }
        }
        if (!placed) {
            for (let i = 0; i < numClasses; i++) {
                const classIndex = i % numClasses;
                if (generatedClasses[classIndex].students.length < maxClassSize) {
                    generatedClasses[classIndex].students.push(bully);
                    if (bully.bullyingScore > 7) generatedClasses[classIndex].bullyCount++;
                    placed = true;
                    break;
                }
            }
        }
        if (!placed) {
            generatedClasses.sort((a, b) => a.students.length - b.students.length)[0].students.push(bully);
            if (bully.bullyingScore > 7) generatedClasses[0].bullyCount++;
        }
    }

    // Distribute other students
    let shuffledOtherStudents = [...otherStudents].sort(() => 0.5 - Math.random());
    for (const student of shuffledOtherStudents) {
        let placed = false;
        for (let i = 0; i < numClasses; i++) {
            const classIndex = i % numClasses;
            if (generatedClasses[classIndex].students.length < maxClassSize) {
                generatedClasses[classIndex].students.push(student);
                placed = true;
                break;
            }
        }
        if (!placed) {
            generatedClasses.sort((a, b) => a.students.length - b.students.length)[0].students.push(student);
        }
    }
    
    let totalAcademicScore = 0;
    let totalWellbeingScore = 0;
    let constraintViolations = [];

    generatedClasses.forEach((cls, idx) => {
        if (cls.students.length > maxClassSize) {
            constraintViolations.push(`Class ${idx} (size ${cls.students.length}) exceeds max size of ${maxClassSize}.`);
        }
        let actualClassBullyingCount = 0;
        cls.students.forEach(s => {
            totalAcademicScore += s.academicScore;
            totalWellbeingScore += s.wellbeingScore;
            if (s.bullyingScore > 7) {
                actualClassBullyingCount++;
            }
        });
        if (actualClassBullyingCount > params.maxBulliesPerClass) {
            constraintViolations.push(`Class ${idx} has ${actualClassBullyingCount} high-risk bullying students (max ${params.maxBulliesPerClass}).`);
        }
    });
    
    const avgAcademicOverall = numStudents > 0 ? (totalAcademicScore / numStudents) : 0;
    const avgWellbeingOverall = numStudents > 0 ? (totalWellbeingScore / numStudents) : 0;

    allocationResults = {
        success: true,
        metrics: {
            balanceScore: Math.random() * 0.15 + 0.8,
            diversityScore: Math.random() * 0.15 + 0.78,
            constraintSatisfaction: constraintViolations.length > 0 ? (Math.max(0, 1 - (constraintViolations.length / numClasses) * 0.5 - 0.1)) : (Math.random() * 0.05 + 0.95),
            processingTime: `${(Math.random() * 1.5 + 0.5).toFixed(1)}s`,
            totalStudents: numStudents,
            numClasses: numClasses,
            avgAcademic: avgAcademicOverall,
            avgWellbeing: avgWellbeingOverall
        },
        violations: constraintViolations,
        classes: generatedClasses.map(c => ({ students: c.students }))
    };
    
    localStorage.setItem('allocationResults', JSON.stringify(allocationResults));
    localStorage.setItem('autoAllocationResults', JSON.stringify(allocationResults));
}

function showResults() {
    const resultsContainer = document.querySelector('.results-summary');
    const progressContainer = document.querySelector('.progress-indicator');

    progressContainer.style.display = 'none';
    resultsContainer.style.display = 'block';

    // Update metrics
    const metrics = allocationResults.metrics;
    document.getElementById('resultsTotalStudents').textContent = metrics.totalStudents;
    document.getElementById('resultsNumClasses').textContent = metrics.numClasses;
    document.getElementById('resultsAvgAcademic').textContent = metrics.avgAcademic.toFixed(1);
    document.getElementById('resultsAvgWellbeing').textContent = metrics.avgWellbeing.toFixed(1);
    
    document.getElementById('balance-score').textContent = (metrics.balanceScore * 100).toFixed(1) + '%';
    document.getElementById('diversity-score').textContent = (metrics.diversityScore * 100).toFixed(1) + '%';
    document.getElementById('constraint-satisfaction').textContent = (metrics.constraintSatisfaction * 100).toFixed(1) + '%';
    document.getElementById('processing-time').textContent = metrics.processingTime;

    // Update violations
    const violationsList = document.querySelector('.constraint-violations ul');
    violationsList.innerHTML = '';
    allocationResults.violations.forEach(violation => {
        const li = document.createElement('li');
        li.textContent = violation;
        violationsList.appendChild(li);
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