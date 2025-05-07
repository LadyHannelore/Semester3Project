// Global state to store the current dataset
let currentDataset = null;

// Required columns for validation
const requiredColumns = [
    'Student_ID',
    'Academic_Performance',
    'Wellbeing_Score',
    'Bullying_Score'
];

// Optional columns
const optionalColumns = {
    social: ['Friends', 'Disrespect'],
    mentalHealth: ['K6_Score', 'Anxiety_Level', 'Depression_Level']
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Handle navigation links
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href && href !== '#') {
                e.preventDefault();
                window.location.href = href;
            }
        });
    });

    initializeFileUpload();
    loadStoredDataset();
});

// File upload initialization
function initializeFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#4299e1';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#cbd5e0';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#cbd5e0';
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'text/csv') {
            handleFileUpload(file);
        } else {
            showValidationResult('Please upload a valid CSV file', false);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });
}

// Handle file upload
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const csvData = e.target.result;
        const rows = csvData.split('\n');
        const headers = rows[0].split(',').map(header => header.trim());
        
        // Store the data
        currentDataset = {
            headers: headers,
            rows: rows.slice(1).map(row => row.split(',').map(cell => cell.trim())),
            source: 'upload'
        };

        // Show preview
        showPreview(currentDataset);
    };
    reader.readAsText(file);
}

// Validate dataset
function validateDataset() {
    if (!currentDataset) {
        showValidationResult('Please upload a dataset first', false);
        return;
    }

    const missingColumns = requiredColumns.filter(col => 
        !currentDataset.headers.includes(col)
    );

    if (missingColumns.length > 0) {
        showValidationResult(`Missing required columns: ${missingColumns.join(', ')}`, false);
        return;
    }

    // Check for empty values in required columns
    const requiredIndices = requiredColumns.map(col => 
        currentDataset.headers.indexOf(col)
    );

    const hasEmptyValues = currentDataset.rows.some(row => 
        requiredIndices.some(index => !row[index])
    );

    if (hasEmptyValues) {
        showValidationResult('Dataset contains empty values in required columns', false);
        return;
    }

    showValidationResult('All fields valid', true);
    updateDatasetSummary(currentDataset);
}

// Show validation result
function showValidationResult(message, isSuccess) {
    const resultDiv = document.getElementById('validationResult');
    resultDiv.className = `validation-result ${isSuccess ? 'validation-success' : 'validation-error'}`;
    resultDiv.innerHTML = `${isSuccess ? '✅' : '❌'} ${message}`;
}

// Show preview table
function showPreview(dataset) {
    const previewDiv = document.getElementById('previewTable');
    if (!dataset || !dataset.rows.length) {
        previewDiv.innerHTML = '';
        return;
    }

    const table = document.createElement('table');
    table.className = 'preview-table';

    // Add headers
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    dataset.headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Add first 10 rows
    const tbody = document.createElement('tbody');
    dataset.rows.slice(0, 10).forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    previewDiv.innerHTML = '';
    previewDiv.appendChild(table);
}

// Generate synthetic data
async function generateSyntheticData() {
    const studentCount = parseInt(document.getElementById('studentCount').value);
    const includeSocial = document.getElementById('includeSocial').checked;
    const includeMentalHealth = document.getElementById('includeMentalHealth').checked;

    try {
        const response = await fetch('http://127.0.0.1:5001/generate', { // Ensure the URL matches the Flask server
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                studentCount,
                includeSocial,
                includeMentalHealth
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Backend error: ${errorText}`);
        }

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || "Synthetic data generation failed.");
        }

        // Update the current dataset with the generated data
        currentDataset = {
            headers: result.headers,
            rows: result.rows,
            source: 'synthetic'
        };

        showPreview(currentDataset);
        updateDatasetSummary(currentDataset);
    } catch (error) {
        showValidationResult(error.message, false);
    }
}

// Update dataset summary
function updateDatasetSummary(dataset) {
    if (!dataset) return;

    const summaryDiv = document.getElementById('datasetSummary');
    const totalRows = dataset.rows.length;
    
    // Calculate metrics
    const academicScores = dataset.rows.map(row => 
        parseFloat(row[dataset.headers.indexOf('Academic_Performance')])
    );
    const wellbeingScores = dataset.rows.map(row => 
        parseFloat(row[dataset.headers.indexOf('Wellbeing_Score')])
    );
    const bullyingScores = dataset.rows.map(row => 
        parseFloat(row[dataset.headers.indexOf('Bullying_Score')])
    );

    const avgAcademic = (academicScores.reduce((a, b) => a + b, 0) / totalRows).toFixed(1);
    const avgWellbeing = (wellbeingScores.reduce((a, b) => a + b, 0) / totalRows).toFixed(1);
    const highRiskBullies = ((bullyingScores.filter(score => score >= 6).length / totalRows) * 100).toFixed(2);

    // Count missing values
    const missingValues = dataset.rows.reduce((count, row) => 
        count + row.filter(cell => !cell).length, 0
    );

    summaryDiv.innerHTML = `
        <div class="summary-card">
            <h4>Total Rows</h4>
            <div class="value">${totalRows}</div>
        </div>
        <div class="summary-card">
            <h4>Missing Values</h4>
            <div class="value">${missingValues}</div>
        </div>
        <div class="summary-card">
            <h4>Avg Academic Score</h4>
            <div class="value">${avgAcademic}</div>
        </div>
        <div class="summary-card">
            <h4>Avg Wellbeing Score</h4>
            <div class="value">${avgWellbeing}</div>
        </div>
        <div class="summary-card">
            <h4>High-Risk Bullies</h4>
            <div class="value">${highRiskBullies}%</div>
        </div>
    `;
}

// Use uploaded data
function useUploadedData() {
    if (!currentDataset || currentDataset.source !== 'upload') {
        showValidationResult('Please upload a dataset first', false);
        return;
    }
    
    validateDataset();
    if (document.querySelector('.validation-success')) {
        // Store in localStorage
        localStorage.setItem('classforgeDataset', JSON.stringify(currentDataset));
        window.location.href = 'student-explorer.html';
    }
}

// Use synthetic data
function useSyntheticData() {
    if (!currentDataset || currentDataset.source !== 'synthetic') {
        showValidationResult('Please generate synthetic data first', false);
        return;
    }
    
    // Store in localStorage
    localStorage.setItem('classforgeDataset', JSON.stringify(currentDataset));
    window.location.href = 'student-explorer.html';
}

// Store dataset in localStorage
function storeDataset() {
    if (currentDataset) {
        localStorage.setItem('classforgeDataset', JSON.stringify(currentDataset));
        return true;
    }
    return false;
}

// Load stored dataset
function loadStoredDataset() {
    const stored = localStorage.getItem('classforgeDataset');
    if (stored) {
        currentDataset = JSON.parse(stored);
        showPreview(currentDataset);
        updateDatasetSummary(currentDataset);
        return true;
    }
    return false;
}

// Navigation function
function navigateTo(page) {
    if (page === 'student-explorer' && !currentDataset) {
        showValidationResult('Please upload or generate data first', false);
        return;
    }
    window.location.href = `${page}.html`;
}