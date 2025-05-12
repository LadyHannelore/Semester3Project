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
    // Implementation will be added later
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
    // Implementation will be added later
    console.log('Processing uploaded file...');
}

// Generate synthetic data
function generateSyntheticData() {
    // Implementation will be added later
    console.log('Generating synthetic data...');
}

// Save data to localStorage
function saveData(data) {
    localStorage.setItem('classforgeDataset', JSON.stringify(data));
    console.log('Data saved to localStorage');
}