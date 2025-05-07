// Global state
let modelResults = [];
let currentChart = null;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadModelResults();
    initializeEventListeners();
    updateCharts('balance'); // Default chart view
});

// Load model results from localStorage
function loadModelResults() {
    const stored = localStorage.getItem('modelResults');
    if (stored) {
        modelResults = JSON.parse(stored);
        updateResultsTable();
        updateModelDropdown();
    } else {
        showError('No model results found. Please run allocations first.');
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // Chart type selection
    document.getElementById('chartTypeSelect').addEventListener('change', (e) => {
        updateCharts(e.target.value);
    });

    // Refresh button
    document.querySelector('.refresh-btn').addEventListener('click', () => {
        loadModelResults();
    });

    // Apply model button
    document.getElementById('applyModelBtn').addEventListener('click', applySelectedModel);

    // Rerun model button
    document.getElementById('rerunModelBtn').addEventListener('click', rerunSelectedModel);
}

// Update results table
function updateResultsTable() {
    const tbody = document.getElementById('modelComparisonBody');
    tbody.innerHTML = '';

    modelResults.forEach((result, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${result.modelName}</td>
            <td>${result.academicStdDev.toFixed(2)}</td>
            <td>${result.wellbeingStdDev.toFixed(2)}</td>
            <td>${result.bullyingViolations}</td>
            <td>${result.classSizeDeviation.toFixed(2)}</td>
            <td>${result.runtime.toFixed(2)}</td>
            <td>${result.totalViolations}</td>
            <td>
                <div class="preferred-toggle ${result.preferred ? 'active' : ''}"
                     onclick="togglePreferred(${index})"></div>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Update model dropdown
function updateModelDropdown() {
    const select = document.getElementById('modelSelect');
    select.innerHTML = '<option value="">Select a model result to use...</option>';

    modelResults.forEach((result, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${result.modelName} (${result.preferred ? 'Preferred' : 'Alternative'})`;
        select.appendChild(option);
    });
}

// Toggle preferred model
function togglePreferred(index) {
    modelResults.forEach((result, i) => {
        result.preferred = (i === index);
    });
    
    // Save to localStorage
    localStorage.setItem('modelResults', JSON.stringify(modelResults));
    
    // Update UI
    updateResultsTable();
    updateModelDropdown();
}

// Update charts based on selected type
function updateCharts(chartType) {
    if (currentChart) {
        currentChart.destroy();
    }

    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    switch (chartType) {
        case 'balance':
            createBalanceChart(ctx);
            break;
        case 'violations':
            createViolationsChart(ctx);
            break;
        case 'radar':
            createRadarChart(ctx);
            break;
    }
}

// Create balance comparison chart
function createBalanceChart(ctx) {
    const labels = modelResults.map(r => r.modelName);
    const academicData = modelResults.map(r => r.academicStdDev);
    const wellbeingData = modelResults.map(r => r.wellbeingStdDev);

    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Academic Balance (StdDev)',
                    data: academicData,
                    backgroundColor: 'rgba(66, 153, 225, 0.5)',
                    borderColor: 'rgb(66, 153, 225)',
                    borderWidth: 1
                },
                {
                    label: 'Wellbeing Balance (StdDev)',
                    data: wellbeingData,
                    backgroundColor: 'rgba(72, 187, 120, 0.5)',
                    borderColor: 'rgb(72, 187, 120)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Standard Deviation'
                    }
                }
            }
        }
    });
}

// Create violations comparison chart
function createViolationsChart(ctx) {
    const labels = modelResults.map(r => r.modelName);
    
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Bullying Violations',
                    data: modelResults.map(r => r.bullyingViolations),
                    borderColor: 'rgb(245, 101, 101)',
                    tension: 0.1
                },
                {
                    label: 'Class Size Violations',
                    data: modelResults.map(r => r.classSizeDeviation),
                    borderColor: 'rgb(66, 153, 225)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Violation Count'
                    }
                }
            }
        }
    });
}

// Create radar chart for fairness comparison
function createRadarChart(ctx) {
    const labels = ['Academic Balance', 'Wellbeing Balance', 'Bullying Control', 'Size Distribution', 'Speed'];
    const datasets = modelResults.map(result => ({
        label: result.modelName,
        data: [
            normalizeScore(result.academicStdDev, 'academicStdDev'),
            normalizeScore(result.wellbeingStdDev, 'wellbeingStdDev'),
            normalizeScore(result.bullyingViolations, 'bullyingViolations'),
            normalizeScore(result.classSizeDeviation, 'classSizeDeviation'),
            normalizeScore(result.runtime, 'runtime')
        ],
        fill: true,
        backgroundColor: result.preferred ? 
            'rgba(66, 153, 225, 0.2)' : 'rgba(160, 174, 192, 0.2)',
        borderColor: result.preferred ? 
            'rgb(66, 153, 225)' : 'rgb(160, 174, 192)',
        pointBackgroundColor: result.preferred ? 
            'rgb(66, 153, 225)' : 'rgb(160, 174, 192)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: result.preferred ? 
            'rgb(66, 153, 225)' : 'rgb(160, 174, 192)'
    }));

    currentChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Normalize scores for radar chart (0 to 1, where 1 is best)
function normalizeScore(value, metric) {
    const allValues = modelResults.map(r => r[metric]);
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    
    // Invert the normalization since lower values are better
    return 1 - ((value - min) / (max - min) || 0);
}

// Apply selected model
function applySelectedModel() {
    const selectedIndex = document.getElementById('modelSelect').value;
    if (!selectedIndex) {
        showError('Please select a model result to apply');
        return;
    }

    const selectedResult = modelResults[selectedIndex];
    localStorage.setItem('allocationResults', JSON.stringify(selectedResult.allocation));
    showSuccess('Model results applied successfully');
}

// Rerun selected model
function rerunSelectedModel() {
    const selectedIndex = document.getElementById('modelSelect').value;
    if (!selectedIndex) {
        showError('Please select a model to re-run');
        return;
    }

    const selectedModel = modelResults[selectedIndex];
    // Redirect to allocation page with selected model
    window.location.href = `allocation.html?model=${encodeURIComponent(selectedModel.modelName)}`;
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