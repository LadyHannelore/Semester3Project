/**
 * History module for ClassForge
 * Handles retrieving and displaying past allocations from the database
 */

// Constants
const API_BASE_URL = 'http://localhost:5001';
const API_ENDPOINTS = {
    ALLOCATIONS: '/api/allocations',
    ALLOCATION_DETAIL: (id) => `/api/allocations/${id}`,
    STUDENTS: '/api/students'
};

/**
 * Load and display allocation history
 */
async function loadAllocationHistory() {
    try {
        showLoading('history-container', 'Loading allocation history...');
        
        // Fetch allocation history from API
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ALLOCATIONS}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch allocation history: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        if (!data.success || !data.allocations) {
            throw new Error('Invalid response format from server');
        }
        
        const allocations = data.allocations;
        
        // Display allocations in the history container
        const historyContainer = document.getElementById('history-container');
        historyContainer.innerHTML = '';
        
        if (allocations.length === 0) {
            historyContainer.innerHTML = `
                <div class="empty-state">
                    <p>No allocation history found.</p>
                    <p>Run an allocation to see results here.</p>
                </div>
            `;
            return;
        }
        
        // Create history items
        allocations.forEach(allocation => {
            const historyItem = createAllocationHistoryItem(allocation);
            historyContainer.appendChild(historyItem);
        });
        
    } catch (error) {
        console.error('Error loading allocation history:', error);
        showError('history-container', `Failed to load allocation history: ${error.message}`);
    }
}

/**
 * Create a history item element for an allocation
 * @param {Object} allocation - The allocation data
 * @returns {HTMLElement} - The history item element
 */
function createAllocationHistoryItem(allocation) {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.setAttribute('data-allocation-id', allocation.id);
    
    // Format date
    const date = new Date(allocation.createdAt);
    const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    
    // Get metrics summary
    const metrics = allocation.metrics || {};
    const classCount = Object.keys(metrics.classSizes || {}).length || 'N/A';
    const avgScore = metrics.averageScore?.toFixed(2) || 'N/A';
    const balanceScore = (metrics.classBalance * 100)?.toFixed(1) || 'N/A';
    
    item.innerHTML = `
        <div class="history-header">
            <h3>Allocation #${allocation.id}</h3>
            <span class="history-date">${formattedDate}</span>
        </div>
        <div class="history-stats">
            <div class="stat">
                <span class="stat-label">Algorithm:</span>
                <span class="stat-value">${formatAlgorithmName(allocation.algorithmType)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Classes:</span>
                <span class="stat-value">${classCount}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Avg. Score:</span>
                <span class="stat-value">${avgScore}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Balance:</span>
                <span class="stat-value">${balanceScore}%</span>
            </div>
        </div>
        <div class="history-actions">
            <button class="btn btn-sm btn-primary view-allocation-btn">
                View Details
            </button>
            <button class="btn btn-sm btn-secondary load-allocation-btn">
                Load Into Classroom
            </button>
        </div>
    `;
    
    // Add event listeners
    item.querySelector('.view-allocation-btn').addEventListener('click', () => {
        viewAllocationDetails(allocation.id);
    });
    
    item.querySelector('.load-allocation-btn').addEventListener('click', () => {
        loadAllocationToClassroom(allocation.id);
    });
    
    return item;
}

/**
 * Format algorithm name for display
 * @param {string} algorithmType - The algorithm type from the API
 * @returns {string} - Formatted algorithm name
 */
function formatAlgorithmName(algorithmType) {
    switch(algorithmType) {
        case 'genetic_algorithm':
            return 'Genetic Algorithm';
        case 'constraint_programming':
            return 'Constraint Programming';
        case 'reinforcement_learning':
            return 'Reinforcement Learning';
        default:
            return algorithmType?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) || 'Unknown';
    }
}

/**
 * View details of a specific allocation
 * @param {number} allocationId - The allocation ID to view
 */
async function viewAllocationDetails(allocationId) {
    try {
        showLoading('allocation-details', 'Loading allocation details...');
        
        // Fetch allocation details from API
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ALLOCATION_DETAIL(allocationId)}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch allocation details: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        if (!data.success || !data.allocation) {
            throw new Error('Invalid response format from server');
        }
        
        const allocation = data.allocation;
        
        // Show allocation details modal
        const detailsContainer = document.getElementById('allocation-details');
        if (!detailsContainer) {
            throw new Error('Allocation details container not found');
        }
        
        // Populate details
        detailsContainer.innerHTML = `
            <h2>Allocation #${allocation.id} Details</h2>
            <div class="details-section">
                <h3>Overview</h3>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="detail-label">Algorithm:</span>
                        <span class="detail-value">${formatAlgorithmName(allocation.algorithmType)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Created:</span>
                        <span class="detail-value">${new Date(allocation.createdAt).toLocaleString()}</span>
                    </div>
                </div>
            </div>
            
            <div class="details-section">
                <h3>Parameters</h3>
                <div class="details-grid">
                    ${renderParametersGrid(allocation.parameters)}
                </div>
            </div>
            
            <div class="details-section">
                <h3>Metrics</h3>
                <div class="details-grid">
                    ${renderMetricsGrid(allocation.metrics)}
                </div>
            </div>
            
            <div class="details-section">
                <h3>Classes</h3>
                <div class="class-summary">
                    ${renderClassSummary(allocation.classes)}
                </div>
            </div>
            
            <div class="modal-actions">
                <button class="btn btn-primary load-allocation-btn">Load Into Classroom</button>
                <button class="btn btn-secondary close-modal-btn">Close</button>
            </div>
        `;
        
        // Add event listeners
        detailsContainer.querySelector('.load-allocation-btn').addEventListener('click', () => {
            loadAllocationToClassroom(allocation.id);
            hideModal('details-modal');
        });
        
        detailsContainer.querySelector('.close-modal-btn').addEventListener('click', () => {
            hideModal('details-modal');
        });
        
        // Show modal
        showModal('details-modal');
        
    } catch (error) {
        console.error('Error loading allocation details:', error);
        showError('allocation-details', `Failed to load allocation details: ${error.message}`);
    }
}

/**
 * Render parameters grid HTML
 * @param {Object} parameters - The allocation parameters
 * @returns {string} - HTML for parameters grid
 */
function renderParametersGrid(parameters) {
    if (!parameters || Object.keys(parameters).length === 0) {
        return '<p>No parameters available</p>';
    }
    
    return Object.entries(parameters).map(([key, value]) => `
        <div class="detail-item">
            <span class="detail-label">${formatParameterName(key)}:</span>
            <span class="detail-value">${value === null ? 'N/A' : value}</span>
        </div>
    `).join('');
}

/**
 * Format parameter name for display
 * @param {string} name - Parameter name
 * @returns {string} - Formatted name
 */
function formatParameterName(name) {
    return name.replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .replace(/([a-z])([A-Z])/g, '$1 $2')
        .replace(/([a-zA-Z])([0-9])/g, '$1 $2')
        .replace(/([0-9])([a-zA-Z])/g, '$1 $2')
        .replace(/([a-z])([A-Z][a-z])/g, '$1 $2');
}

/**
 * Render metrics grid HTML
 * @param {Object} metrics - The allocation metrics
 * @returns {string} - HTML for metrics grid
 */
function renderMetricsGrid(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) {
        return '<p>No metrics available</p>';
    }
    
    return Object.entries(metrics).map(([key, value]) => {
        // Special handling for complex metrics
        if (typeof value === 'object' && value !== null) {
            return '';  // Skip complex objects for now
        }
        
        // Format numeric values
        let displayValue = value;
        if (typeof value === 'number') {
            if (key.toLowerCase().includes('percentage') || key.toLowerCase().includes('balance') || key.toLowerCase().includes('score')) {
                displayValue = (value * 100).toFixed(1) + '%';
            } else {
                displayValue = value.toFixed(2);
            }
        }
        
        return `
            <div class="detail-item">
                <span class="detail-label">${formatParameterName(key)}:</span>
                <span class="detail-value">${displayValue}</span>
            </div>
        `;
    }).join('');
}

/**
 * Render class summary HTML
 * @param {Object} classes - The classes object with student IDs
 * @returns {string} - HTML for class summary
 */
function renderClassSummary(classes) {
    if (!classes || Object.keys(classes).length === 0) {
        return '<p>No class data available</p>';
    }
    
    const classCount = Object.keys(classes).length;
    const totalStudents = Object.values(classes).reduce((sum, students) => sum + students.length, 0);
    const avgClassSize = (totalStudents / classCount).toFixed(1);
    
    let html = `
        <p><strong>${classCount}</strong> classes with <strong>${totalStudents}</strong> total students (avg. ${avgClassSize} per class)</p>
        <div class="class-distribution">
    `;
    
    // Add class distribution bars
    Object.entries(classes).forEach(([classId, students]) => {
        const percentage = (students.length / totalStudents * 100).toFixed(1);
        html += `
            <div class="class-bar">
                <div class="class-bar-label">Class ${classId}</div>
                <div class="class-bar-container">
                    <div class="class-bar-fill" style="width: ${percentage}%"></div>
                    <div class="class-bar-text">${students.length} students</div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    return html;
}

/**
 * Load allocation data into the classroom view
 * @param {number} allocationId - The allocation ID to load
 */
async function loadAllocationToClassroom(allocationId) {
    try {
        showMessage('Loading allocation into classroom view...');
        
        // Fetch allocation details
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.ALLOCATION_DETAIL(allocationId)}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch allocation: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        if (!data.success || !data.allocation) {
            throw new Error('Invalid response format from server');
        }
        
        // Store allocation in local storage for classroom view
        localStorage.setItem('currentAllocation', JSON.stringify(data.allocation));
        
        // Navigate to classroom view
        window.location.href = 'classroom.html';
        
    } catch (error) {
        console.error('Error loading allocation to classroom:', error);
        showError('message-container', `Failed to load allocation: ${error.message}`);
    }
}

/**
 * Show loading message in a container
 * @param {string} containerId - The ID of the container
 * @param {string} message - The loading message
 */
function showLoading(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
    }
}

/**
 * Show error message in a container
 * @param {string} containerId - The ID of the container
 * @param {string} message - The error message
 */
function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="error-message">
                <p>${message}</p>
            </div>
        `;
    }
}

/**
 * Show a message toast
 * @param {string} message - The message to display
 */
function showMessage(message) {
    const messageContainer = document.getElementById('message-container');
    if (!messageContainer) {
        const container = document.createElement('div');
        container.id = 'message-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = 'message-toast';
    toast.textContent = message;
    
    document.getElementById('message-container').appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => {
            toast.remove();
        }, 500);
    }, 3000);
}

/**
 * Show a modal dialog
 * @param {string} modalId - The ID of the modal element
 */
function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('show');
    }
}

/**
 * Hide a modal dialog
 * @param {string} modalId - The ID of the modal element
 */
function hideModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('show');
    }
}

// Add event listeners when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const historyTab = document.getElementById('history-tab');
    if (historyTab) {
        loadAllocationHistory();
    }
    
    // Add global event listeners for modals
    document.querySelectorAll('.modal-overlay').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('show');
            }
        });
    });
});
