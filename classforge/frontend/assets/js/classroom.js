// Performance constants
const DEBOUNCE_DELAY = 300; // ms for search input
const THROTTLE_DELAY = 16; // ms for 60fps animations
const CACHE_DURATION = 300000; // 5 minutes
const ANIMATION_DURATION = 200; // ms for transitions

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

// Cache for expensive operations
const classroomCache = new Map();

// Global state
let allocationData = null;
let currentClassId = null;
let currentFilters = {
    academic: { min: null, max: null },
    wellbeing: { min: null, max: null },
    flags: {
        highRisk: false,
        needsAttention: false,
        bullying: false
    }
};
let sortConfig = {
    column: null,
    direction: 'asc'
};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    console.log("Classroom.js DOMContentLoaded");
    initializePage();
    initializeEventListeners();
});

function initializePage() {
    console.log("Initializing classroom page...");
    loadAllocationData();
    if (allocationData && allocationData.classes && allocationData.classes.length > 0) {
        populateClassList();
        if (allocationData.classes[0]) {
            const firstClassId = String(allocationData.classes[0].classId);
            console.log("Attempting to render first class by default:", firstClassId);
            setTimeout(() => {
                const firstClassItem = document.querySelector(`.class-item[data-class-id="${firstClassId}"]`);
                if (firstClassItem) {
                    firstClassItem.click();
                } else {
                    console.warn("First class item not found in DOM for default selection.");
                }
            }, 0);
        }
    } else {
        console.warn("No allocation data or no classes found to initialize page.");
        displayNoAllocationDataMessage();
    }
}

function displayNoAllocationDataMessage() {
    const classDetailsSection = document.querySelector('.class-details');
    if (classDetailsSection) {
        classDetailsSection.innerHTML = `
            <div class="no-data-message" style="padding: 2rem; text-align: center; background: #fff; border-radius: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="font-size: 1.5rem; color: #2d3748;">No Classroom Data Available</h3>
                <p style="color: #718096; margin-top: 1rem;">
                    It seems no class allocation has been run yet, or the data could not be loaded.
                </p>
                <p style="margin-top: 0.5rem;">
                    Please go to the <a href="allocation.html" style="color: #4299e1; text-decoration: underline;">Group Allocation</a> page to run an allocation.
                </p>
            </div>`;
    }
    const classListContainer = document.getElementById('classList');
    if (classListContainer) {
        classListContainer.innerHTML = '<p style="padding: 1rem; text-align: center; color: #718096;">No classes found.</p>';
    }
}

function loadAllocationData() {
    const storedData = localStorage.getItem('allocationResults');
    console.log("Attempting to load 'allocationResults' from localStorage.");
    if (storedData) {
        try {
            allocationData = JSON.parse(storedData);
            console.log("Loaded allocationData:", JSON.parse(JSON.stringify(allocationData)));
            if (!allocationData || !allocationData.classes || !Array.isArray(allocationData.classes)) {
                console.error("Allocation data is malformed:", allocationData);
                allocationData = null;
            } else if (allocationData.classes.length === 0) {
                console.warn("Allocation data loaded, but no classes found within.");
            } else {
                console.log(`Successfully loaded ${allocationData.classes.length} classes.`);
            }
        } catch (e) {
            console.error("Error parsing allocation data from localStorage:", e);
            allocationData = null;
        }
    } else {
        console.warn("No 'allocationResults' data found in localStorage.");
        allocationData = null;
    }
}

function populateClassList() {
    console.log("Populating class list...");
    const classListContainer = document.getElementById('classList');
    if (!classListContainer) {
        console.error("Class list container not found.");
        return;
    }
    classListContainer.innerHTML = '';

    if (!allocationData || !allocationData.classes || allocationData.classes.length === 0) {
        console.warn("No classes to populate in the list.");
        classListContainer.innerHTML = '<p style="padding:1rem; text-align:center; color:#718096;">No classes available.</p>';
        return;
    }

    allocationData.classes.forEach(cls => {
        const classItem = document.createElement('div');
        classItem.className = 'class-item';
        classItem.dataset.classId = cls.classId;
        classItem.setAttribute('role', 'button');
        classItem.setAttribute('tabindex', '0');
        classItem.innerHTML = `
            <span class="class-icon">🏫</span>
            <div class="class-info">
                <span class="class-name">Class ${cls.classId}</span>
                <span class="class-stats">${cls.students ? cls.students.length : 0} students</span>
            </div>
        `;
        classItem.addEventListener('click', () => {
            console.log(`Class item clicked: ${cls.classId}`);
            renderClassroomView(String(cls.classId));
            document.querySelectorAll('.class-item.active').forEach(active => active.classList.remove('active'));
            classItem.classList.add('active');
        });
        classListContainer.appendChild(classItem);
    });
    console.log("Class list populated.");
}

function renderClassroomView(classId) {
    console.log(`Rendering classroom view for classId: ${classId}`);
    currentClassId = classId;
    if (!allocationData || !allocationData.classes) {
        console.error("Cannot render classroom view: allocationData is missing.");
        displayNoAllocationDataMessage();
        return;
    }

    const selectedClassData = allocationData.classes.find(c => String(c.classId) === String(classId));
    console.log(`Selected class data for classId ${classId}:`, selectedClassData ? JSON.parse(JSON.stringify(selectedClassData)) : 'Not found');

    const classDetailsSection = document.querySelector('.class-details');
    if (!classDetailsSection) {
        console.error("Class details section not found in DOM.");
        return;
    }
    classDetailsSection.innerHTML = `
        <section class="class-summary">
        </section>
        <section class="students-table" aria-label="Students in Class">
        </section>
    `;

    if (selectedClassData) {
        renderClassSummary(selectedClassData);
        renderStudentsTable(selectedClassData.students || []);
        if (selectedClassData.students && selectedClassData.students.length > 0) {
            console.log("Calling renderClassroomCharts and renderClassSocialGraphs");
            renderClassroomCharts(selectedClassData);
            renderClassSocialGraphs(selectedClassData);
        } else {
            console.warn(`No students in class ${classId} to render charts or social graphs.`);
            const chartsContainer = document.getElementById('classDistributionCharts');
            if (chartsContainer) chartsContainer.innerHTML = '<p class="chart-empty-message" style="text-align:center; width:100%;">No student data for distribution charts.</p>';
            const socialChartsContainer = document.getElementById('classSocialCharts');
            if (socialChartsContainer) socialChartsContainer.innerHTML = '<p class="chart-empty-message" style="text-align:center; width:100%;">No student data for social graph.</p>';
        }
    } else {
        console.warn(`No data found for classId: ${classId}. Clearing class details.`);
        classDetailsSection.innerHTML = `<p style="padding:2rem; text-align:center;">Details for class ${classId} not found.</p>`;
    }
}

function renderClassSummary(classData) {
    const summarySection = document.querySelector('.class-summary');
    if (!summarySection) return;
    summarySection.innerHTML = `
        <div class="summary-header">
            <h3>Class ${classData.classId} Summary</h3>
            <span>${classData.students ? classData.students.length : 0} Students</span>
        </div>
        <div class="classroom-charts" id="classDistributionCharts">
            <div class="chart-container" id="classAcademicChartContainer"></div>
            <div class="chart-container" id="classWellbeingChartContainer"></div>
        </div>
        <div class="classroom-social-charts" id="classSocialCharts">
            <div class="social-chart-container">
                <h4>Friends Network</h4>
                <canvas id="classFriendsGraph" width="600" height="600"></canvas>
            </div>
        </div>
    `;
    console.log(`Rendered summary for class ${classData.classId}`);
}

function renderStudentsTable(students) {
    const tableSection = document.querySelector('.students-table');
    if (!tableSection) return;
    tableSection.innerHTML = `
        <div class="table-header"><h3>Student List</h3></div>
        <div class="table-container"><table><thead><tr><th>ID</th><th>Academic</th><th>Wellbeing</th></tr></thead><tbody>
        ${students.map(s => `<tr><td>${s.id}</td><td>${s.academicScore}</td><td>${s.wellbeingScore}</td></tr>`).join('')}
        </tbody></table></div>`;
    if (students.length === 0) {
        tableSection.querySelector('tbody').innerHTML = '<tr><td colspan="3" style="text-align:center;">No students in this class.</td></tr>';
    }
    console.log(`Rendered student table with ${students.length} students.`);
}

function initializeEventListeners() {
    // Use debounced search for better performance
    const classSearch = document.getElementById('classSearch');
    if (classSearch) {
        classSearch.addEventListener('input', debounce(filterClassList, DEBOUNCE_DELAY));
    }
    
    const studentSearch = document.getElementById('studentSearch');
    if (studentSearch) {
        studentSearch.addEventListener('input', debounce(filterStudentTable, DEBOUNCE_DELAY));
    }
    
    const filterBtn = document.getElementById('filterBtn');
    if (filterBtn) {
        filterBtn.addEventListener('click', showFilterModal);
    }
    
    // Use throttled sorting for better responsiveness
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', throttle(() => sortTable(th.dataset.sort), THROTTLE_DELAY));
    });
}

let classAcademicChart = null;
let classWellbeingChart = null;

function renderClassroomCharts(classData) {
    console.log("Rendering classroom charts for classId:", classData.classId, "Students:", classData.students ? classData.students.length : 0);
    if (!classData.students || classData.students.length === 0) {
        console.warn("No students to render classroom charts.");
        const academicContainer = document.getElementById('classAcademicChartContainer');
        if (academicContainer) academicContainer.innerHTML = '<p class="chart-empty-message">No data for Academic Chart.</p>';
        const wellbeingContainer = document.getElementById('classWellbeingChartContainer');
        if (wellbeingContainer) wellbeingContainer.innerHTML = '<p class="chart-empty-message">No data for Wellbeing Chart.</p>';
        return;
    }

    const chartOptionsBase = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { 
                beginAtZero: true, 
                title: { display: true, text: 'Number of Students', font: { size: 18 } }, // Increased
                ticks: { font: { size: 14 } } // Increased
            },
            x: { 
                title: { display: true, text: 'Score Range', font: { size: 18 } }, // Increased
                ticks: { font: { size: 14 } } // Increased
            }
        },
        plugins: {
            legend: { display: true, labels: { font: { size: 16 } } }, // Increased
            title: { display: true, font: { size: 20, weight: 'bold' }, padding: { top: 10, bottom: 15 } } // Increased
        }
    };

    const academicScores = classData.students.map(s => s.academicScore).filter(score => typeof score === 'number' && !isNaN(score));
    console.log("Academic scores for chart:", JSON.parse(JSON.stringify(academicScores)));
    const academicCtx = getOrCreateChartCanvas('classAcademicChartContainer', 'classAcademicChart');
    if (academicCtx) {
        if (classAcademicChart) classAcademicChart.destroy();
        if (academicScores.length > 0) {
            const academicDistribution = calculateDistribution(academicScores, 0, 100, 10);
            console.log("Academic distribution:", JSON.parse(JSON.stringify(academicDistribution)));
            classAcademicChart = new Chart(academicCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(academicDistribution),
                    datasets: [{ label: 'Academic Performance', data: Object.values(academicDistribution), backgroundColor: '#4299e1' }]
                },
                options: { ...chartOptionsBase, plugins: { ...chartOptionsBase.plugins, title: { ...chartOptionsBase.plugins.title, text: 'Class Academic Performance'}}}
            });
        } else {
            academicCtx.canvas.parentElement.innerHTML = '<p class="chart-empty-message">No valid academic scores for this class.</p>';
        }
    }

    const wellbeingScores = classData.students.map(s => s.wellbeingScore).filter(score => typeof score === 'number' && !isNaN(score));
    console.log("Wellbeing scores for chart:", JSON.parse(JSON.stringify(wellbeingScores)));
    const wellbeingCtx = getOrCreateChartCanvas('classWellbeingChartContainer', 'classWellbeingChart');
    if (wellbeingCtx) {
        if (classWellbeingChart) classWellbeingChart.destroy();
        if (wellbeingScores.length > 0) {
            const wellbeingDistribution = calculateDistribution(wellbeingScores, 0, 10, 1);
            console.log("Wellbeing distribution:", JSON.parse(JSON.stringify(wellbeingDistribution)));
            classWellbeingChart = new Chart(wellbeingCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(wellbeingDistribution),
                    datasets: [{ label: 'Wellbeing Score', data: Object.values(wellbeingDistribution), backgroundColor: '#48bb78' }]
                },
                options: { ...chartOptionsBase, plugins: { ...chartOptionsBase.plugins, title: { ...chartOptionsBase.plugins.title, text: 'Class Wellbeing Scores'}}}
            });
        } else {
             wellbeingCtx.canvas.parentElement.innerHTML = '<p class="chart-empty-message">No valid wellbeing scores for this class.</p>';
        }
    }
}

function getOrCreateChartCanvas(containerId, canvasId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Chart container ${containerId} not found.`);
        return null;
    }
    container.innerHTML = '';

    let canvas = document.createElement('canvas');
    canvas.id = canvasId;
    canvas.width = 600; 
    canvas.height = 600; 
    container.appendChild(canvas);
    
    console.log(`Canvas ${canvasId} created or found in ${containerId}.`);
    return canvas.getContext('2d');
}

function calculateDistribution(scoresArray, min, max, step) {
    const distribution = {};
    for (let i = min; i < max; i += step) {
        const upperBinBound = (i + step -1 >= max && i < max) ? max : i + step -1;
        const rangeLabel = `${i}-${upperBinBound}`;
        distribution[rangeLabel] = 0;
    }
    
    let validScoresCount = 0;
    scoresArray.forEach(score => {
        const val = parseFloat(score); 
        if (isNaN(val)) {
            return; 
        }
        validScoresCount++;

        let foundBin = false;
        for (let i = min; i < max; i += step) {
            const upperBinBoundExclusive = i + step;
            const upperBinBoundInclusiveLabel = (i + step -1 >= max && i < max) ? max : i + step -1;
            const rangeLabel = `${i}-${upperBinBoundInclusiveLabel}`;

            if (val >= i && val < upperBinBoundExclusive) {
                 if (distribution[rangeLabel] !== undefined) {
                    distribution[rangeLabel]++;
                 } else {
                    console.warn(`Bin label ${rangeLabel} not found for value ${val}. Check bin initialization.`);
                 }
                foundBin = true;
                break;
            }
        }
        if (!foundBin && val === max) {
            const lastBinStart = Math.floor((max - 1) / step) * step;
             const lastBinUpperLabel = (lastBinStart + step -1 >= max && lastBinStart < max) ? max : lastBinStart + step -1;
            const rangeLabel = `${lastBinStart}-${lastBinUpperLabel}`;
            if (distribution[rangeLabel] !== undefined) {
                 distribution[rangeLabel]++;
            } else {
                 console.warn(`Last bin label ${rangeLabel} for max value ${val} not found.`);
            }
        }
    });
    return distribution;
}

function renderClassSocialGraphs(classData) {
    console.log("Rendering social graphs for classId:", classData.classId);
    if (!classData.students || classData.students.length === 0) {
        console.warn("No students to render social graphs.");
        const socialContainer = document.getElementById('classSocialCharts');
        if (socialContainer) socialContainer.innerHTML = '<p class="chart-empty-message" style="text-align:center; width:100%;">No student data for social graph.</p>';
        return;
    }
    const friendsCanvas = document.getElementById('classFriendsGraph');
    if (friendsCanvas) {
        const friendsCtx = friendsCanvas.getContext('2d');
        console.log("Student data for Friends Network:", JSON.parse(JSON.stringify(classData.students.slice(0,5))));
        renderSocialAdjacencyMatrix(friendsCtx, classData.students, 'friends', 'Friends Network');
    } else {
        console.warn("Canvas for classFriendsGraph not found.");
    }
}

function renderSocialAdjacencyMatrix(ctx, students, relationKey, label) {
    console.log(`[Classroom] Rendering social graph: ${label} for relationKey: ${relationKey} with ${students.length} students.`);
    if (!students || students.length === 0) {
        console.warn(`No students data for graph: ${label}`);
        ctx.font = "16px Inter, sans-serif";
        ctx.fillStyle = "#718096";
        ctx.textAlign = "center";
        ctx.fillText(`No student data for ${label}.`, ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }

    const validStudents = students.filter(s => s && s.id !== undefined && s.id !== null && String(s.id).trim() !== "");
    if (validStudents.length !== students.length) {
        console.warn(`Filtered out ${students.length - validStudents.length} students with invalid IDs for graph ${label}`);
    }
    if (validStudents.length === 0) {
        console.warn(`No valid students after filtering for graph: ${label}`);
        ctx.font = "16px Inter, sans-serif"; ctx.fillStyle = "#718096"; ctx.textAlign = "center";
        ctx.fillText(`No valid student data for ${label}.`, ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    }

    const ids = validStudents.map(s => String(s.id));
    const idToIdx = Object.fromEntries(ids.map((id, i) => [id, i]));
    
    const radius = Math.min(ctx.canvas.width, ctx.canvas.height) / 2 - 70;
    const centerX = ctx.canvas.width / 2;
    const centerY = ctx.canvas.height / 2;
    const nodeRadius = 18;

    const numNodes = validStudents.length;
    const nodePositions = [];
    for (let i = 0; i < numNodes; i++) {
        const angle = (i / numNodes) * 2 * Math.PI;
        nodePositions.push({
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle),
            id: ids[i]
        });
    }

    ctx.strokeStyle = "#b0bdcc";
    ctx.lineWidth = 1.5;
    validStudents.forEach((student, i) => {
        let friendsArray = [];
        const rawFriendsData = student[relationKey]; 

        if (typeof rawFriendsData === 'string' && rawFriendsData.trim() !== '') {
            friendsArray = rawFriendsData.split(',')
                                .map(idStr => String(idStr.trim()))
                                .filter(idStr => idStr !== '');
        } else if (Array.isArray(rawFriendsData)) {
            friendsArray = rawFriendsData.map(id => String(id).trim()).filter(idStr => idStr !== '');
        }

        const p1 = nodePositions[i];
        friendsArray.forEach(friendId => {
            if (friendId in idToIdx) {
                const friendIndex = idToIdx[friendId];
                if (friendIndex < numNodes) { 
                    const p2 = nodePositions[friendIndex];
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            }
        });
    });

    ctx.font = "14px Inter, sans-serif"; // Increased font size for node labels
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    nodePositions.forEach(pos => {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius, 0, 2 * Math.PI);
        ctx.fillStyle = "#4299e1"; 
        ctx.fill();
        ctx.strokeStyle = "#2a6299";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "#ffffff"; 
        ctx.fillText(pos.id, pos.x, pos.y);
    });

    ctx.font = "bold 20px Inter, sans-serif"; // Increased title font size
    ctx.fillStyle = "#1a202c";
    ctx.textAlign = "center";
    ctx.fillText(label, centerX, 40);
}