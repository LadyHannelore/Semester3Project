// Performance constants
const DEBOUNCE_DELAY = 300; // ms for parameter changes
const THROTTLE_DELAY = 16; // ms for progress updates
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB max file size
const CHUNK_SIZE = 1024 * 1024; // 1MB chunks for large file processing

// Utility functions for performance optimization
function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

function throttle(func, delay) {
    let lastCall = 0;
    return function (...args) {
        const now = new Date().getTime();
        if (now - lastCall < delay) return;
        lastCall = now;
        return func.apply(this, args);
    };
}

// File validation helper
function validateFile(file) {
    const errors = [];
    
    if (!file) {
        errors.push('No file selected');
        return errors;
    }
    
    if (file.size > MAX_FILE_SIZE) {
        errors.push(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (${MAX_FILE_SIZE / 1024 / 1024}MB)`);
    }
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        errors.push('File must be a CSV (.csv) file');
    }
    
    return errors;
}

// Global state
let uploadedData = null;
let dataPreviewTable = null;
let generationParameters = {
    studentCount: 100,
    meanAcademic: 70,
    stdAcademic: 15,
    meanWellbeing: 6.5,
    stdWellbeing: 1.5,
    bullyingPercent: 10,
    friendsPerStudent: 3
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

// Set up event listeners for file upload with validation
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

// Initialize generation controls with debounced parameter updates
function initializeGenerationControls() {
    // Update generationParameters when inputs change with debouncing
    const paramIds = [
        'studentCount', 'meanAcademic', 'stdAcademic',
        'meanWellbeing', 'stdWellbeing', 'bullyingPercent', 'friendsPerStudent'
    ];
    
    paramIds.forEach(id => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', debounce(function() {
                const value = parseFloat(input.value);
                if (!isNaN(value)) {
                    generationParameters[id] = value;
                    validateGenerationParameters();
                }
            }, DEBOUNCE_DELAY));
        }
    });
    console.log('Generation controls initialized');
}

// Add validation function for generation parameters
function validateGenerationParameters() {
    const errors = [];
    
    if (generationParameters.studentCount < 10 || generationParameters.studentCount > 10000) {
        errors.push('Student count must be between 10 and 10,000');
    }
    
    if (generationParameters.meanAcademic < 0 || generationParameters.meanAcademic > 100) {
        errors.push('Mean academic score must be between 0 and 100');
    }
    
    if (generationParameters.stdAcademic < 1 || generationParameters.stdAcademic > 50) {
        errors.push('Academic standard deviation must be between 1 and 50');
    }
    
    if (generationParameters.meanWellbeing < 1 || generationParameters.meanWellbeing > 10) {
        errors.push('Mean wellbeing score must be between 1 and 10');
    }
    
    if (generationParameters.stdWellbeing < 0.1 || generationParameters.stdWellbeing > 5) {
        errors.push('Wellbeing standard deviation must be between 0.1 and 5');
    }
    
    if (generationParameters.bullyingPercent < 0 || generationParameters.bullyingPercent > 50) {
        errors.push('Bullying percentage must be between 0 and 50');
    }
    
    if (generationParameters.friendsPerStudent < 0 || generationParameters.friendsPerStudent > 20) {
        errors.push('Friends per student must be between 0 and 20');
    }
    
    // Display validation errors
    const errorContainer = document.getElementById('generation-errors');
    if (errorContainer) {
        if (errors.length > 0) {
            errorContainer.innerHTML = `<div class="error-message">${errors.join('<br>')}</div>`;
            errorContainer.style.display = 'block';
        } else {
            errorContainer.style.display = 'none';
        }
    }
    
    return errors.length === 0;
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
        bullyingPercent: parseFloat(document.getElementById('bullyingPercent').value),
        friendsPerStudent: parseInt(document.getElementById('friendsPerStudent')?.value, 10) || 3
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
    // Assign friends (as comma-separated StudentIDs)
    students.forEach((student, idx) => {
        const possibleFriends = students.map(s => s.StudentID).filter(id => id !== student.StudentID);
        const numFriends = Math.min(params.friendsPerStudent, possibleFriends.length);
        const friends = [];
        while (friends.length < numFriends) {
            const pick = possibleFriends[Math.floor(Math.random() * possibleFriends.length)];
            if (!friends.includes(pick)) friends.push(pick);
        }
        student.Friends = friends.join(',');
    });

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
    if (previewSection) {
        previewSection.style.display = 'block';
    }

    // Create or update the data table
    if (!dataPreviewTable) {
        dataPreviewTable = new DataTable('#data-preview-table', {
            data: data,
            columns: [
                { title: 'Student ID', data: 'StudentID' },
                { title: 'Academic Performance', data: 'Academic_Performance' },
                { title: 'Wellbeing Score', data: 'Wellbeing_Score' },
                { title: 'Bullying Score', data: 'Bullying_Score' },
                { title: 'Friends', data: 'Friends' }
            ],
            pageLength: 10,
            searchable: true,
            order: [[1, 'desc']]
        });
    } else {
        dataPreviewTable.clear().rows.add(data).draw();
    }
}

//# sourceMappingURL=upload.js.map