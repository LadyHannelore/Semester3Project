// Global state
let uploadedData = null;
let dataPreviewTable = null;
let generationParameters = {
    studentCount: 100,
    meanAcademic: 70,
    stdAcademic: 15,
    meanWellbeing: 6.5,
    stdWellbeing: 1.5,
    bullyingPercent: 10
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadListeners();
    initializeGenerationControls();
    // Add event listener for generate button
    const generateBtn = document.getElementById('generateBtn');
    if (generateBtn) {
        generateBtn.addEventListener('click', function(e) {
            e.preventDefault();
            generateSyntheticData();
        });
    }
});

// Set up event listeners for file upload
function initializeUploadListeners() {
    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }
    
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            processUploadedFile();
        });
    }
}

// Initialize generation controls
function initializeGenerationControls() {
    // Optionally, update generationParameters when inputs change
    const paramIds = [
        'studentCount', 'meanAcademic', 'stdAcademic',
        'meanWellbeing', 'stdWellbeing', 'bullyingPercent'
    ];
    paramIds.forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', function() {
                generationParameters[id] = parseFloat(input.value);
            });
        }
    });
    console.log('Generation controls initialized');
}

// Handle file upload event
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // Show file details
        const fileDetails = document.getElementById('file-details');
        if (fileDetails) {
            fileDetails.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
            fileDetails.style.display = 'block';
        }
    }
}

// Format file size to human-readable format
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' bytes';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / 1048576).toFixed(1) + ' MB';
}

// Process the uploaded file
function processUploadedFile() {
    console.log('Processing uploaded file...');
}

// Generate synthetic data
function generateSyntheticData() {
    // Read parameters from input fields
    const params = {
        studentCount: parseInt(document.getElementById('studentCount').value, 10),
        meanAcademic: parseFloat(document.getElementById('meanAcademic').value),
        stdAcademic: parseFloat(document.getElementById('stdAcademic').value),
        meanWellbeing: parseFloat(document.getElementById('meanWellbeing').value),
        stdWellbeing: parseFloat(document.getElementById('stdWellbeing').value),
        bullyingPercent: parseFloat(document.getElementById('bullyingPercent').value)
    };

    // Generate synthetic data
    const students = [];
    for (let i = 0; i < params.studentCount; i++) {
        const academicScore = Math.max(0, Math.min(100, Math.round(randomNormal(params.meanAcademic, params.stdAcademic))));
        const wellbeingScore = Math.max(0, Math.min(10, +(randomNormal(params.meanWellbeing, params.stdWellbeing)).toFixed(2)));
        const bullyingScore = (Math.random() < params.bullyingPercent / 100)
            ? Math.floor(Math.random() * 4) + 7 // 7-10 for bullies
            : Math.floor(Math.random() * 7);    // 0-6 for non-bullies
        students.push({
            StudentID: i + 1,
            Academic_Performance: academicScore,
            Wellbeing_Score: wellbeingScore,
            Bullying_Score: bullyingScore
        });
    }
    uploadedData = students;
    previewData(students);
}

// Helper: random normal distribution
function randomNormal(mean, std) {
    // Box-Muller transform
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Preview data in the table
function previewData(data) {
    if (!data || data.length === 0) return;
    // Show preview section
    const previewSection = document.querySelector('.data-preview-section');
    if (previewSection) previewSection.style.display = 'block';

    // Set headers
    const headers = Object.keys(data[0]);
    const headerRow = document.getElementById('preview-headers');
    headerRow.innerHTML = '';
    headers.forEach(h => {
        const th = document.createElement('th');
        th.textContent = h;
        headerRow.appendChild(th);
    });

    // Set body
    const body = document.getElementById('preview-body');
    body.innerHTML = '';
    data.slice(0, 20).forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(h => {
            const td = document.createElement('td');
            td.textContent = row[h];
            tr.appendChild(td);
        });
        body.appendChild(tr);
    });

    // Wire up "Use This Data" button
    const useDataBtn = document.getElementById('useDataBtn');
    if (useDataBtn) {
        useDataBtn.onclick = function() {
            saveData(data);
            alert('Synthetic data saved! You can now proceed.');
        };
    }
}

// Save data to localStorage
function saveData(data) {
    // Convert array of objects to { headers: [...], rows: [...] }
    if (!Array.isArray(data) || data.length === 0) return;
    const headers = Object.keys(data[0]);
    const rows = data.map(row => headers.map(h => row[h]));
    const dataset = { headers, rows };
    localStorage.setItem('classforgeDataset', JSON.stringify(dataset));
    console.log('Data saved to localStorage');
}