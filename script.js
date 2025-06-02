document.addEventListener('DOMContentLoaded', function() {
    loadDataAndRenderCharts();
    initializeResponsiveCharts();
});

// Performance constants
const CHART_ANIMATION_DURATION = 300; // ms for chart animations
const RESIZE_DEBOUNCE_DELAY = 250; // ms for window resize events
const CHART_UPDATE_THROTTLE = 100; // ms for chart updates
const CACHE_DURATION = 300000; // 5 minutes

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

// Chart cache for better performance
const chartCache = new Map();

// Optimized chart configuration
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
        duration: CHART_ANIMATION_DURATION
    },
    plugins: {
        legend: {
            display: true,
            position: 'top'
        },
        tooltip: {
            enabled: true,
            intersect: false,
            mode: 'nearest'
        }
    }
};

// Initialize responsive chart handling
function initializeResponsiveCharts() {
    const debouncedResize = debounce(() => {
        // Resize all charts when window resizes
        Object.values(Chart.instances).forEach(chart => {
            if (chart) {
                chart.resize();
            }
        });
    }, RESIZE_DEBOUNCE_DELAY);
    
    window.addEventListener('resize', debouncedResize);
}

// Function to load data and render all charts on the dashboard
function loadDataAndRenderCharts() {
    const storedDataJSON = localStorage.getItem('classforgeDataset');
    if (!storedDataJSON) {
        console.warn("No data found in localStorage for index charts (classforgeDataset).");
        const chartGrid = document.querySelector('.charts-grid');
        if (chartGrid) {
            chartGrid.innerHTML = '<p style="text-align:center; grid-column: 1 / -1; padding: 2rem; font-size: 1.1rem; color: #718096;">No student data available to display overview charts. Please go to "Upload & Simulate" to load or generate data.</p>';
        }
        ['schoolAcademicChartContainer', 'schoolWellbeingChartContainer', 'schoolBullyingChartContainer', 'schoolFriendsChartContainer'].forEach(id => {
            const container = document.getElementById(id);
            if (container) container.style.display = 'none';
        });
        return;
    }

    try {
        const dataset = JSON.parse(storedDataJSON);
        console.log("Dataset loaded for index.html charts:", dataset);

        if (!dataset || !dataset.headers || !dataset.rows || !Array.isArray(dataset.headers) || !Array.isArray(dataset.rows)) {
            console.error("Stored dataset (classforgeDataset) is not in the expected format {headers: [], rows: []}.", dataset);
            const chartGrid = document.querySelector('.charts-grid');
            if (chartGrid) {
                chartGrid.innerHTML = '<p style="text-align:center; grid-column: 1 / -1; padding: 2rem; font-size: 1.1rem; color: #c53030;">Error: Invalid data format in localStorage.</p>';
            }
            return;
        }

        const { headers, rows: studentRows } = dataset;

        if (studentRows.length === 0) {
            console.warn("No student rows in the dataset for index.html charts.");
            const chartGrid = document.querySelector('.charts-grid');
            if (chartGrid) {
                chartGrid.innerHTML = '<p style="text-align:center; grid-column: 1 / -1; padding: 2rem; font-size: 1.1rem; color: #718096;">Dataset is empty. No charts to display.</p>';
            }
            return;
        }

        const academicCanvas = document.getElementById('schoolAcademicChart');
        const academicColName = 'Academic_Performance';
        const academicColIndex = headers.indexOf(academicColName);
        if (academicCanvas && academicColIndex !== -1) {
            const academicData = studentRows.map(row => parseFloat(row[academicColIndex])).filter(val => !isNaN(val));
            console.log("Academic Data for Histogram:", academicData);
            if (academicData.length > 0) {
                renderHistogram(academicCanvas, academicData, 'Academic Performance', 'rgba(54, 162, 235, 0.5)', 0, 100, 10);
            } else {
                console.warn("No valid academic data to render histogram.");
                academicCanvas.parentElement.innerHTML += '<p class="chart-empty-message">No academic data.</p>';
            }
        } else {
            console.warn(`Academic Performance column ('${academicColName}') not found or canvas missing.`);
            if (academicCanvas) academicCanvas.parentElement.innerHTML += `<p class="chart-empty-message">Academic Performance data column not found.</p>`;
        }

        const wellbeingCanvas = document.getElementById('schoolWellbeingChart');
        const wellbeingColName = 'Wellbeing_Score';
        const wellbeingColIndex = headers.indexOf(wellbeingColName);
        if (wellbeingCanvas && wellbeingColIndex !== -1) {
            const wellbeingData = studentRows.map(row => parseFloat(row[wellbeingColIndex])).filter(val => !isNaN(val));
            console.log("Wellbeing Data for Histogram:", wellbeingData);
            if (wellbeingData.length > 0) {
                renderHistogram(wellbeingCanvas, wellbeingData, 'Wellbeing Score', 'rgba(75, 192, 192, 0.5)', 0, 10, 1);
            } else {
                console.warn("No valid wellbeing data to render histogram.");
                wellbeingCanvas.parentElement.innerHTML += '<p class="chart-empty-message">No wellbeing data.</p>';
            }
        } else {
            console.warn(`Wellbeing Score column ('${wellbeingColName}') not found or canvas missing.`);
            if (wellbeingCanvas) wellbeingCanvas.parentElement.innerHTML += `<p class="chart-empty-message">Wellbeing Score data column not found.</p>`;
        }

        const bullyingCanvas = document.getElementById('schoolBullyingChart');
        const bullyingColName = 'Bullying_Score';
        const bullyingColIndex = headers.indexOf(bullyingColName);
        if (bullyingCanvas && bullyingColIndex !== -1) {
            const bullyingData = studentRows.map(row => parseFloat(row[bullyingColIndex])).filter(val => !isNaN(val));
            console.log("Bullying Data for Histogram:", bullyingData);
            if (bullyingData.length > 0) {
                renderHistogram(bullyingCanvas, bullyingData, 'Bullying Score', 'rgba(255, 99, 132, 0.5)', 0, 10, 1);
            } else {
                console.warn("No valid bullying data to render histogram.");
                bullyingCanvas.parentElement.innerHTML += '<p class="chart-empty-message">No bullying data.</p>';
            }
        } else {
            console.warn(`Bullying Score column ('${bullyingColName}') not found or canvas missing.`);
            if (bullyingCanvas) bullyingCanvas.parentElement.innerHTML += `<p class="chart-empty-message">Bullying Score data column not found.</p>`;
        }

        const friendsGraphCanvas = document.getElementById('schoolFriendsGraph');
        const studentIdColName = 'StudentID';
        const friendsColName = 'Friends';
        const studentIdColIdx = headers.indexOf(studentIdColName);
        const friendsColIdx = headers.indexOf(friendsColName);

        if (friendsGraphCanvas && studentIdColIdx !== -1 && friendsColIdx !== -1) {
            console.log("Data for Friends Graph:", studentRows.slice(0, 5));
            renderSchoolFriendsGraph(friendsGraphCanvas, studentRows, studentIdColIdx, friendsColIdx, 'Schoolwide Friends Network');
        } else {
            console.warn(`Required columns for Friends Graph ('${studentIdColName}', '${friendsColName}') not found or canvas missing.`);
            if (friendsGraphCanvas) friendsGraphCanvas.parentElement.innerHTML += `<p class="chart-empty-message">Friends data columns not found.</p>`;
        }

    } catch (error) {
        console.error("Error parsing or processing stored dataset for index charts:", error);
        const chartGrid = document.querySelector('.charts-grid');
        if (chartGrid) {
            chartGrid.innerHTML = `<p style="text-align:center; grid-column: 1 / -1; padding: 2rem; font-size: 1.1rem; color: #c53030;">Error loading chart data: ${error.message}</p>`;
        }
    }
}

function calculateDistribution(data, min, max, step) {
    const distribution = {};
    for (let i = min; i < max; i += step) {
        const rangeLabel = `${i}-${i + step - 1}`;
        distribution[rangeLabel] = 0;
    }

    let validScoresCount = 0;
    data.forEach(value => {
        const val = parseFloat(value);
        if (isNaN(val)) {
            return;
        }
        validScoresCount++;

        let foundBin = false;
        for (let i = min; i < max; i += step) {
            if (val >= i && val < (i + step)) {
                const rangeLabel = `${i}-${i + step - 1}`;
                distribution[rangeLabel]++;
                foundBin = true;
                break;
            }
        }
        if (!foundBin && val === max) {
            const lastBinStart = max - step;
            const rangeLabel = `${lastBinStart}-${lastBinStart + step - 1}`;
            if (distribution[rangeLabel] !== undefined) {
                distribution[rangeLabel]++;
            }
        }
    });
    return distribution;
}

// Optimized histogram rendering with caching
function renderHistogram(canvas, data, label, color, min, max, step) {
    if (!canvas || !data || data.length === 0) {
        if (canvas && canvas.parentElement) {
            const existingMsg = canvas.parentElement.querySelector('.chart-empty-message');
            if (existingMsg) existingMsg.remove();
            canvas.parentElement.innerHTML += `<p class="chart-empty-message">No data available for ${label}.</p>`;
        }
        return;
    }
    
    // Create cache key for this chart
    const cacheKey = `histogram_${label}_${data.length}_${min}_${max}_${step}`;
    
    // Check if we have cached distribution data
    let bins;
    if (chartCache.has(cacheKey)) {
        const cached = chartCache.get(cacheKey);
        if (Date.now() - cached.timestamp < CACHE_DURATION) {
            bins = cached.data;
        } else {
            chartCache.delete(cacheKey);
        }
    }
    
    if (!bins) {
        bins = calculateDistribution(data, min, max, step);
        // Cache the calculated distribution
        chartCache.set(cacheKey, {
            data: bins,
            timestamp: Date.now()
        });
    }

    const ctx = canvas.getContext('2d');

    const chartDataValues = Object.values(bins);
    if (chartDataValues.every(v => v === 0)) {
        console.warn(`All bins are zero for histogram "${label}". Chart will appear empty.`);
        if (canvas.parentElement) {
            const existingMsg = canvas.parentElement.querySelector('.chart-empty-message');
            if (existingMsg) existingMsg.remove();
            canvas.parentElement.innerHTML += `<p class="chart-empty-message">Data available, but all distribution bins are zero for ${label}.</p>`;
        }
    }

    if (ctx._chartInstance) {
        ctx._chartInstance.destroy();
    }
    
    // Use optimized chart configuration
    const chartConfig = {
        type: 'bar',
        data: {
            labels: Object.keys(bins),
            datasets: [{
                label: label,
                data: Object.values(bins),
                backgroundColor: color,
                borderColor: color.replace('0.2', '1').replace('0.5', '1'),
                borderWidth: 1.5
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Number of Students', font: { size: 18 } },
                    ticks: { font: { size: 14 } }
                },
                x: {
                    title: { display: true, text: label, font: { size: 18 } },
                    ticks: { font: { size: 14 } }
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                legend: {
                    display: true,
                    labels: { font: { size: 16 } }
                },
                title: {
                    display: true,
                    text: label,
                    font: { size: 20, weight: 'bold' },
                    padding: { top: 10, bottom: 20 }
                }
            }
        }
    };
    
    ctx._chartInstance = new Chart(ctx, chartConfig);
}

function renderSchoolFriendsGraph(canvas, studentRows, studentIdColIndex, friendsColIndex, graphLabel) {
    if (!canvas) {
        console.warn("Canvas not found for school friends graph.");
        return;
    }
    const ctx = canvas.getContext('2d');

    if (ctx._chartInstance) {
        ctx._chartInstance.destroy();
        ctx._chartInstance = null;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!studentRows || studentRows.length === 0) {
        console.warn("No student data for school friends graph.");
        ctx.font = "16px Inter, sans-serif"; ctx.fillStyle = "#718096"; ctx.textAlign = "center";
        ctx.fillText("No student data for friends graph.", canvas.width / 2, canvas.height / 2);
        return;
    }
    console.log(`[script.js] Rendering school friends graph with ${studentRows.length} students.`);

    const students = studentRows.map(row => ({
        id: String(row[studentIdColIndex]),
        friends: row[friendsColIndex] || ""
    })).filter(s => s.id && s.id !== "undefined" && s.id !== "null");

    if (students.length === 0) {
        console.warn("No valid students after filtering for school friends graph.");
        ctx.font = "16px Inter, sans-serif"; ctx.fillStyle = "#718096"; ctx.textAlign = "center";
        ctx.fillText("No valid student data for graph.", canvas.width / 2, canvas.height / 2);
        return;
    }

    const ids = students.map(s => s.id);
    const idToIdx = Object.fromEntries(ids.map((id, i) => [id, i]));

    const numNodes = students.length;
    const maxNodesToDisplay = 100;
    const displayNodes = numNodes > maxNodesToDisplay ? students.slice(0, maxNodesToDisplay) : students;
    const displayIds = displayNodes.map(s => s.id);
    const displayIdToIdx = Object.fromEntries(displayIds.map((id, i) => [id, i]));
    const numDisplayNodes = displayNodes.length;

    const radius = Math.min(ctx.canvas.width, ctx.canvas.height) / 2 - (numDisplayNodes > 50 ? 40 : 60);
    const centerX = ctx.canvas.width / 2;
    const centerY = ctx.canvas.height / 2;
    const nodeRadius = numDisplayNodes > 50 ? 10 : 15;

    const nodePositions = [];
    for (let i = 0; i < numDisplayNodes; i++) {
        const angle = (i / numDisplayNodes) * 2 * Math.PI;
        nodePositions.push({
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle),
            id: displayIds[i]
        });
    }

    ctx.strokeStyle = "#b8c2cc";
    ctx.lineWidth = 1;
    displayNodes.forEach((student, i) => {
        let friendsArray = [];
        const rawFriendsData = student.friends;

        if (typeof rawFriendsData === 'string' && rawFriendsData.trim() !== '') {
            friendsArray = rawFriendsData.split(',')
                                .map(idStr => String(idStr.trim()))
                                .filter(idStr => idStr !== '');
        } else if (Array.isArray(rawFriendsData)) {
            friendsArray = rawFriendsData.map(id => String(id).trim()).filter(idStr => idStr !== '');
        }

        const p1 = nodePositions[i];
        friendsArray.forEach(friendId => {
            if (friendId in displayIdToIdx) {
                const friendIndex = displayIdToIdx[friendId];
                const p2 = nodePositions[friendIndex];
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }
        });
    });

    ctx.font = (numDisplayNodes > 50 ? "12px" : "14px") + " Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    nodePositions.forEach(pos => {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
        ctx.fillStyle = "#fbbf24";
        ctx.fill();
        ctx.strokeStyle = "#d69e2e";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = "#1a202c";
        if (numDisplayNodes <= 50) {
            ctx.fillText(pos.id, pos.x, pos.y);
        }
    });

    ctx.font = "bold 20px Inter, sans-serif";
    ctx.fillStyle = "#1a202c";
    ctx.textAlign = "center";
    ctx.fillText(graphLabel + (numNodes > maxNodesToDisplay ? ` (showing ${maxNodesToDisplay} of ${numNodes} students)` : ""), centerX, 40);
}