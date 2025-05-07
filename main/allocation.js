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
    const strategyCards = document.querySelectorAll('.strategy-card input[type="radio"]');
    strategyCards.forEach(card => {
        card.addEventListener('change', function() {
            selectedStrategy = this.value;
            updateParameterVisibility();
            updateRunButtonState();
        });
    });
}

// Parameter controls
function initializeParameterControls() {
    // KMeans parameters
    const kmeansSliders = document.querySelectorAll('#kmeans-params input[type="range"]');
    kmeansSliders.forEach(slider => {
        slider.addEventListener('input', function() {
            updateSliderValue(this);
        });
    });

    // CP-SAT parameters
    const cpSatInputs = document.querySelectorAll('#cpsat-params input[type="number"]');
    cpSatInputs.forEach(input => {
        input.addEventListener('change', validateNumberInput);
    });

    // Hybrid parameters
    const hybridTabs = document.querySelectorAll('.tab-btn');
    hybridTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });
}

// Update parameter visibility based on selected strategy
function updateParameterVisibility() {
    const parameterGroups = document.querySelectorAll('.parameter-group');
    parameterGroups.forEach(group => {
        group.style.display = 'none';
    });

    if (selectedStrategy) {
        const selectedGroup = document.getElementById(`${selectedStrategy}-params`);
        if (selectedGroup) {
            selectedGroup.style.display = 'grid';
        }
    }
}

// Update slider value display
function updateSliderValue(slider) {
    const valueDisplay = slider.parentElement.querySelector('.slider-value');
    if (valueDisplay) {
        valueDisplay.textContent = slider.value;
    }
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

// Switch between hybrid parameter tabs
function switchTab(tabId) {
    const tabs = document.querySelectorAll('.tab-btn');
    const panes = document.querySelectorAll('.tab-pane');

    tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });

    panes.forEach(pane => {
        pane.classList.toggle('active', pane.id === tabId);
    });
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
        // Simulate allocation process
        await simulateAllocation();
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

async function simulateAllocation() {
    const progressFill = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    const steps = [
        { progress: 20, text: 'Analyzing student data...' },
        { progress: 40, text: 'Calculating optimal clusters...' },
        { progress: 60, text: 'Applying constraints...' },
        { progress: 80, text: 'Optimizing assignments...' },
        { progress: 100, text: 'Finalizing results...' }
    ];

    for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        progressFill.style.width = `${step.progress}%`;
        progressText.textContent = step.text;
    }

    // Simulate allocation results
    allocationResults = {
        success: true,
        metrics: {
            balanceScore: 0.85,
            diversityScore: 0.92,
            constraintSatisfaction: 0.95,
            processingTime: '2.3s'
        },
        violations: [
            'Gender balance constraint slightly violated in Group 3',
            'Learning style diversity threshold not met in Group 7'
        ]
    };
}

function showResults() {
    const resultsContainer = document.querySelector('.results-summary');
    const progressContainer = document.querySelector('.progress-indicator');

    progressContainer.style.display = 'none';
    resultsContainer.style.display = 'block';

    // Update metrics
    const metrics = allocationResults.metrics;
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
            window.location.href = 'classroom.html';
            break;
        case 'compare-models':
            // Implement comparison functionality
            console.log('Compare models clicked');
            break;
    }
} 