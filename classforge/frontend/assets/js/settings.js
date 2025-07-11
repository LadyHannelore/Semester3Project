// Performance constants
const DEBOUNCE_DELAY = 300; // ms for input changes
const THROTTLE_DELAY = 16; // ms for slider updates
const VALIDATION_DELAY = 150; // ms for form validation
const SAVE_DELAY = 500; // ms for auto-save

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

// Auto-save functionality
const autoSave = debounce(() => {
    saveSettings();
    showToast('Settings auto-saved', 'success');
}, SAVE_DELAY);

// Default settings
const defaultSettings = {
    classConfig: {
        maxClassSize: 25,
        totalClasses: 'auto',
        manualClassCount: 5,
        allowUneven: false
    },
    constraints: {
        academicThreshold: 1.0,
        wellbeingThreshold: 1.0,
        maxBullies: 2,
        maxFriendsSplit: 2
    },
    visualization: {
        showAcademic: true,
        showNetwork: true,
        showBullying: true
    },
    experimental: {
        enableRL: false,
        enableGNN: false,
        enableLiveReopt: false
    }
};

// Global state
let currentSettings = { ...defaultSettings };

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadSettings();
    initializeEventListeners();
    updateUI();
});

// Load settings from localStorage
function loadSettings() {
    const stored = localStorage.getItem('classforgeSettings');
    if (stored) {
        currentSettings = JSON.parse(stored);
    }
}

// Initialize event listeners with optimized handlers
function initializeEventListeners() {
    // Class Configuration with debounced input handling
    const maxClassSize = document.getElementById('maxClassSize');
    if (maxClassSize) {
        maxClassSize.addEventListener('input', debounce(handleClassConfigChange, DEBOUNCE_DELAY));
    }
    
    const totalClasses = document.getElementById('totalClasses');
    if (totalClasses) {
        totalClasses.addEventListener('change', handleTotalClassesChange);
    }
    
    const manualClassCount = document.getElementById('manualClassCount');
    if (manualClassCount) {
        manualClassCount.addEventListener('input', debounce(handleClassConfigChange, DEBOUNCE_DELAY));
    }
    
    const allowUneven = document.getElementById('allowUneven');
    if (allowUneven) {
        allowUneven.addEventListener('change', handleClassConfigChange);
    }

    // Constraint Parameters with throttled slider updates
    const academicThreshold = document.getElementById('academicThreshold');
    if (academicThreshold) {
        academicThreshold.addEventListener('input', throttle(handleConstraintChange, THROTTLE_DELAY));
    }
    
    const wellbeingThreshold = document.getElementById('wellbeingThreshold');
    if (wellbeingThreshold) {
        wellbeingThreshold.addEventListener('input', throttle(handleConstraintChange, THROTTLE_DELAY));
    }
    
    const maxBullies = document.getElementById('maxBullies');
    if (maxBullies) {
        maxBullies.addEventListener('input', debounce(handleConstraintChange, DEBOUNCE_DELAY));
    }
    
    const maxFriendsSplit = document.getElementById('maxFriendsSplit');
    if (maxFriendsSplit) {
        maxFriendsSplit.addEventListener('input', debounce(handleConstraintChange, DEBOUNCE_DELAY));
    }

    // Visualization Defaults with immediate updates
    const showAcademic = document.getElementById('showAcademic');
    if (showAcademic) {
        showAcademic.addEventListener('change', handleVisualizationChange);
    }
    
    const showNetwork = document.getElementById('showNetwork');
    if (showNetwork) {
        showNetwork.addEventListener('change', handleVisualizationChange);
    }
    
    const showBullying = document.getElementById('showBullying');
    if (showBullying) {
        showBullying.addEventListener('change', handleVisualizationChange);
    }

    // Experimental Features with validation
    const enableRL = document.getElementById('enableRL');
    if (enableRL) {
        enableRL.addEventListener('change', debounce(handleExperimentalChange, VALIDATION_DELAY));
    }
    
    const enableGNN = document.getElementById('enableGNN');
    if (enableGNN) {
        enableGNN.addEventListener('change', debounce(handleExperimentalChange, VALIDATION_DELAY));
    }
    
    const enableLiveReopt = document.getElementById('enableLiveReopt');
    if (enableLiveReopt) {
        enableLiveReopt.addEventListener('change', debounce(handleExperimentalChange, VALIDATION_DELAY));
    }

    // Action Buttons
    const saveSettingsBtn = document.getElementById('saveSettingsBtn');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', saveSettings);
    }
    document.getElementById('resetSettingsBtn').addEventListener('click', resetSettings);

    // Range input value displays
    document.getElementById('academicThreshold').addEventListener('input', updateRangeValue);
    document.getElementById('wellbeingThreshold').addEventListener('input', updateRangeValue);
}

// Update UI with current settings
function updateUI() {
    // Class Configuration
    document.getElementById('maxClassSize').value = currentSettings.classConfig.maxClassSize;
    document.getElementById('totalClasses').value = currentSettings.classConfig.totalClasses;
    document.getElementById('manualClassCount').value = currentSettings.classConfig.manualClassCount;
    document.getElementById('manualClassCount').disabled = currentSettings.classConfig.totalClasses === 'auto';
    document.getElementById('allowUneven').checked = currentSettings.classConfig.allowUneven;

    // Constraint Parameters
    document.getElementById('academicThreshold').value = currentSettings.constraints.academicThreshold;
    document.getElementById('wellbeingThreshold').value = currentSettings.constraints.wellbeingThreshold;
    document.getElementById('maxBullies').value = currentSettings.constraints.maxBullies;
    document.getElementById('maxFriendsSplit').value = currentSettings.constraints.maxFriendsSplit;

    // Update range value displays
    updateRangeValue({ target: document.getElementById('academicThreshold') });
    updateRangeValue({ target: document.getElementById('wellbeingThreshold') });

    // Visualization Defaults
    document.getElementById('showAcademic').checked = currentSettings.visualization.showAcademic;
    document.getElementById('showNetwork').checked = currentSettings.visualization.showNetwork;
    document.getElementById('showBullying').checked = currentSettings.visualization.showBullying;

    // Experimental Features
    document.getElementById('enableRL').checked = currentSettings.experimental.enableRL;
    document.getElementById('enableGNN').checked = currentSettings.experimental.enableGNN;
    document.getElementById('enableLiveReopt').checked = currentSettings.experimental.enableLiveReopt;
}

// Event Handlers
function handleClassConfigChange(event) {
    const id = event.target.id;
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;

    if (id === 'totalClasses') {
        document.getElementById('manualClassCount').disabled = value === 'auto';
    }
    currentSettings.classConfig[id] = value;
    validateSettings();
}

function handleConstraintChange(event) {
    const id = event.target.id;
    const value = parseFloat(event.target.value);
    currentSettings.constraints[id] = value;
    validateSettings();
}

function handleVisualizationChange(event) {
    const id = event.target.id;
    const value = event.target.checked;
    currentSettings.visualization[id] = value;
}

function handleExperimentalChange(event) {
    const id = event.target.id;
    const value = event.target.checked;
    currentSettings.experimental[id] = value;

    if (value && id === 'enableRL') {
        showWarning('Reinforcement Learning feature is in alpha stage and may impact system performance.');
    } else if (value && id === 'enableGNN') {
        showWarning('GNN-based clustering is in alpha stage and requires additional computational resources.');
    }
}

// Update range value displays for sliders
function updateRangeValue(event) {
    const input = event.target;
    const display = input.nextElementSibling;
    if (display) {
        display.textContent = `${input.value} σ`;
    }
}

// Save settings
function saveSettings() {
    if (validateSettings()) {
        localStorage.setItem('classforgeSettings', JSON.stringify(currentSettings));
        broadcastSettingsChange();
        showSuccess('Settings saved successfully');
    }
}

// Reset settings
function resetSettings() {
    if (confirm('Are you sure you want to reset all settings to default values?')) {
        currentSettings = { ...defaultSettings };
        updateUI();
        localStorage.setItem('classforgeSettings', JSON.stringify(currentSettings));
        broadcastSettingsChange();
        showSuccess('Settings reset to defaults');
    }
}

// Validate settings
function validateSettings() {
    let isValid = true;
    const errors = [];

    // Validate class size
    if (currentSettings.classConfig.maxClassSize < 15 || currentSettings.classConfig.maxClassSize > 40) {
        errors.push('Class size must be between 15 and 40 students');
        isValid = false;
    }

    // Validate manual class count
    if (currentSettings.classConfig.totalClasses === 'manual') {
        if (currentSettings.classConfig.manualClassCount < 1 || currentSettings.classConfig.manualClassCount > 20) {
            errors.push('Number of classes must be between 1 and 20');
            isValid = false;
        }
    }

    // Validate thresholds
    if (currentSettings.constraints.academicThreshold < 0 || currentSettings.constraints.academicThreshold > 2) {
        errors.push('Academic threshold must be between 0 and 2');
        isValid = false;
    }

    if (currentSettings.constraints.wellbeingThreshold < 0 || currentSettings.constraints.wellbeingThreshold > 2) {
        errors.push('Wellbeing threshold must be between 0 and 2');
        isValid = false;
    }

    // Show errors if any
    if (!isValid) {
        showError(errors.join('\n'));
    }

    return isValid;
}

// Broadcast settings change to other pages
function broadcastSettingsChange() {
    // Use localStorage event to notify other pages
    localStorage.setItem('settingsUpdated', Date.now().toString());
}

// Show success message (replace alert with a notification system for production)
function showSuccess(message) {
    // TODO: Replace alert with a professional notification system
    alert(message);
}

// Show error message (replace alert with a notification system for production)
function showError(message) {
    // TODO: Replace alert with a professional notification system
    alert('Error: ' + message);
}

// Show warning message (replace alert with a notification system for production)
function showWarning(message) {
    // TODO: Replace alert with a professional notification system
    alert('Warning: ' + message);
}