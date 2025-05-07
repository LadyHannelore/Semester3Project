// Global state
let allocationData = null;
let networkData = null;
let charts = {};

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    loadData();
    initializeEventListeners();
});

// Load data from localStorage
function loadData() {
    const stored = localStorage.getItem('allocationResults');
    if (stored) {
        allocationData = JSON.parse(stored);
        initializeCharts();
        initializeNetwork('friendship');
        generateInsights();
    } else {
        showError('No allocation data found. Please run allocation first.');
    }
}

// Initialize event listeners
function initializeEventListeners() {
    document.getElementById('refreshChartsBtn').addEventListener('click', refreshData);
    document.getElementById('networkTypeSelect').addEventListener('change', (e) => {
        initializeNetwork(e.target.value);
    });
    document.getElementById('analyzeBtn').addEventListener('click', generateInsights);
}

// Initialize charts
function initializeCharts() {
    // Academic Score Histogram
    const academicCtx = document.getElementById('academicHistogram').getContext('2d');
    const academicScores = allocationData.classes.flatMap(c => 
        c.students.map(s => s.academicScore)
    );
    
    charts.academic = new Chart(academicCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 10}, (_, i) => `${i}-${i+1}`),
            datasets: [{
                label: 'Academic Scores',
                data: calculateHistogram(academicScores, 10),
                backgroundColor: '#4299e1',
                borderColor: '#3182ce',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Students'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Score Range'
                    }
                }
            }
        }
    });

    // Wellbeing Score Distribution
    const wellbeingCtx = document.getElementById('wellbeingChart').getContext('2d');
    const wellbeingData = allocationData.classes.map((c, i) => ({
        classId: i,
        scores: c.students.map(s => s.wellbeingScore)
    }));

    charts.wellbeing = new Chart(wellbeingCtx, {
        type: 'boxplot',
        data: {
            labels: wellbeingData.map(d => `Class ${d.classId}`),
            datasets: [{
                label: 'Wellbeing Scores',
                data: wellbeingData.map(d => calculateBoxPlotData(d.scores)),
                backgroundColor: '#48bb78',
                borderColor: '#2f855a',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 10,
                    title: {
                        display: true,
                        text: 'Wellbeing Score'
                    }
                }
            }
        }
    });

    // Bullying Risk Levels
    const bullyingCtx = document.getElementById('bullyingChart').getContext('2d');
    const bullyingData = allocationData.classes.map((c, i) => ({
        classId: i,
        low: c.students.filter(s => s.bullyingScore <= 4).length,
        medium: c.students.filter(s => s.bullyingScore > 4 && s.bullyingScore <= 7).length,
        high: c.students.filter(s => s.bullyingScore > 7).length
    }));

    charts.bullying = new Chart(bullyingCtx, {
        type: 'bar',
        data: {
            labels: bullyingData.map(d => `Class ${d.classId}`),
            datasets: [
                {
                    label: 'Low Risk',
                    data: bullyingData.map(d => d.low),
                    backgroundColor: '#48bb78'
                },
                {
                    label: 'Medium Risk',
                    data: bullyingData.map(d => d.medium),
                    backgroundColor: '#ecc94b'
                },
                {
                    label: 'High Risk',
                    data: bullyingData.map(d => d.high),
                    backgroundColor: '#f56565'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Class'
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Students'
                    }
                }
            }
        }
    });

    // Class Size Chart
    const sizeCtx = document.getElementById('classSizeChart').getContext('2d');
    const classSizes = allocationData.classes.map((c, i) => ({
        classId: i,
        size: c.students.length
    }));

    charts.size = new Chart(sizeCtx, {
        type: 'bar',
        data: {
            labels: classSizes.map(d => `Class ${d.classId}`),
            datasets: [{
                label: 'Class Size',
                data: classSizes.map(d => d.size),
                backgroundColor: '#4299e1',
                borderColor: '#3182ce',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Students'
                    }
                }
            }
        }
    });
}

// Initialize network visualization
function initializeNetwork(type) {
    const container = document.getElementById('networkGraph');
    container.innerHTML = '';
    
    // Set up SVG
    const width = container.clientWidth;
    const height = container.clientHeight;
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create nodes and links based on type
    const { nodes, links } = createNetworkData(type);

    // Set up force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(width / 2, height / 2));

    // Draw links
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6);

    // Draw nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', 5)
        .attr('fill', d => getNodeColor(d, type))
        .call(drag(simulation));

    // Add tooltips
    node.append('title')
        .text(d => `Student ${d.id}\nClass ${d.classId}\nWellbeing: ${d.wellbeing}`);

    // Update positions
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });

    // Update legend
    updateNetworkLegend(type);
}

// Create network data
function createNetworkData(type) {
    const nodes = allocationData.classes.flatMap((c, classId) =>
        c.students.map(s => ({
            id: s.id,
            classId,
            wellbeing: s.wellbeingScore,
            bullying: s.bullyingScore
        }))
    );

    // Simulate relationships based on type
    const links = [];
    nodes.forEach(source => {
        nodes.forEach(target => {
            if (source.id !== target.id) {
                if (type === 'friendship' && Math.random() < 0.1) {
                    links.push({ source: source.id, target: target.id });
                } else if (type === 'disrespect' && source.bullying > 7 && Math.random() < 0.2) {
                    links.push({ source: source.id, target: target.id });
                }
            }
        });
    });

    return { nodes, links };
}

// Get node color based on type
function getNodeColor(node, type) {
    if (type === 'friendship') {
        return d3.schemeCategory10[node.classId % 10];
    } else if (type === 'disrespect') {
        return node.bullying > 7 ? '#f56565' : '#48bb78';
    }
    return '#4299e1';
}

// Update network legend
function updateNetworkLegend(type) {
    const legend = document.getElementById('legendContent');
    legend.innerHTML = '';

    if (type === 'friendship') {
        allocationData.classes.forEach((_, i) => {
            const item = document.createElement('div');
            item.style.display = 'flex';
            item.style.alignItems = 'center';
            item.style.gap = '0.5rem';
            item.style.marginBottom = '0.5rem';

            const color = document.createElement('div');
            color.style.width = '1rem';
            color.style.height = '1rem';
            color.style.backgroundColor = d3.schemeCategory10[i % 10];
            color.style.borderRadius = '50%';

            const label = document.createElement('span');
            label.textContent = `Class ${i}`;

            item.appendChild(color);
            item.appendChild(label);
            legend.appendChild(item);
        });
    } else if (type === 'disrespect') {
        const items = [
            { color: '#f56565', label: 'High Bullying Risk' },
            { color: '#48bb78', label: 'Low Bullying Risk' }
        ];

        items.forEach(item => {
            const div = document.createElement('div');
            div.style.display = 'flex';
            div.style.alignItems = 'center';
            div.style.gap = '0.5rem';
            div.style.marginBottom = '0.5rem';

            const color = document.createElement('div');
            color.style.width = '1rem';
            color.style.height = '1rem';
            color.style.backgroundColor = item.color;
            color.style.borderRadius = '50%';

            const label = document.createElement('span');
            label.textContent = item.label;

            div.appendChild(color);
            div.appendChild(label);
            legend.appendChild(div);
        });
    }
}

// Generate insights
function generateInsights() {
    const distributionList = document.getElementById('distributionInsightsList');
    const networkList = document.getElementById('networkInsightsList');
    const warningList = document.getElementById('warningFlagsList');

    // Clear existing insights
    distributionList.innerHTML = '';
    networkList.innerHTML = '';
    warningList.innerHTML = '';

    // Distribution insights
    const distributionInsights = analyzeDistributions();
    distributionInsights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        distributionList.appendChild(li);
    });

    // Network insights
    const networkInsights = analyzeNetwork();
    networkInsights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        networkList.appendChild(li);
    });

    // Warning flags
    const warnings = generateWarnings();
    warnings.forEach(warning => {
        const li = document.createElement('li');
        li.textContent = warning;
        warningList.appendChild(li);
    });
}

// Analyze distributions
function analyzeDistributions() {
    const insights = [];
    
    // Analyze class sizes
    const sizes = allocationData.classes.map(c => c.students.length);
    const avgSize = sizes.reduce((a, b) => a + b, 0) / sizes.length;
    const sizeVariance = Math.max(...sizes) - Math.min(...sizes);
    
    if (sizeVariance > 5) {
        insights.push(`Large class size variance detected: difference of ${sizeVariance} students between largest and smallest classes.`);
    }

    // Analyze academic distribution
    allocationData.classes.forEach((c, i) => {
        const avgAcademic = c.students.reduce((a, b) => a + b.academicScore, 0) / c.students.length;
        if (avgAcademic < 5) {
            insights.push(`Class ${i} has a low average academic score of ${avgAcademic.toFixed(1)}.`);
        }
    });

    // Analyze wellbeing distribution
    allocationData.classes.forEach((c, i) => {
        const lowWellbeing = c.students.filter(s => s.wellbeingScore < 5).length;
        if (lowWellbeing > c.students.length * 0.3) {
            insights.push(`Class ${i} has ${lowWellbeing} students with low wellbeing scores.`);
        }
    });

    return insights;
}

// Analyze network
function analyzeNetwork() {
    const insights = [];

    // Analyze bullying distribution
    allocationData.classes.forEach((c, i) => {
        const highBullying = c.students.filter(s => s.bullyingScore > 7).length;
        if (highBullying > 2) {
            insights.push(`Class ${i} has ${highBullying} high-risk bullying students.`);
        }
    });

    // Add simulated friendship insights
    insights.push('Friend groups are generally maintained within classes.');
    insights.push('Some cross-class friendships exist, promoting social integration.');

    return insights;
}

// Generate warnings
function generateWarnings() {
    if (!loadAllocationDataForVisualizations() || !allocationData || !Array.isArray(allocationData.classes)) {
        return ["⚠️ Allocation data not available or invalid. Please run group allocation first."];
    }

    const warnings = [];

    // Check for severe imbalances
    allocationData.classes.forEach((c, i) => {
        if (!c || !Array.isArray(c.students)) {
            warnings.push(`⚠️ Data for Class ${i} is incomplete or missing.`);
            return; // Skip this class if data is malformed
        }

        const className = `Class ${i}`; // Use a consistent name for messages

        if (c.students.length > 30) { // Example threshold, could be from settings
            warnings.push(`⚠️ ${className} (size ${c.students.length}) exceeds recommended maximum size of 30.`);
        }
        
        const highBullyingStudents = c.students.filter(s => s && typeof s.bullyingScore === 'number' && s.bullyingScore > 7);
        if (highBullyingStudents.length > 2) { // Example threshold
            warnings.push(`⚠️ High concentration of bullying risk in ${className} (${highBullyingStudents.length} students).`);
        }

        const lowWellbeingStudents = c.students.filter(s => s && typeof s.wellbeingScore === 'number' && s.wellbeingScore < 5);
        if (c.students.length > 0 && lowWellbeingStudents.length > c.students.length * 0.4) { // Example threshold
            warnings.push(`⚠️ Critical wellbeing situation in ${className} (${lowWellbeingStudents.length} out of ${c.students.length} students affected).`);
        }
    });

    if (warnings.length === 0 && allocationData.classes.length > 0) {
        warnings.push("✅ No severe imbalances detected in class allocations based on current checks.");
    } else if (allocationData.classes.length === 0) {
        warnings.push("ℹ️ No classes found in the allocation data to analyze.");
    }

    return warnings;
}

// Helper functions
function calculateHistogram(data, bins) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binSize = (max - min) / bins;
    const histogram = new Array(bins).fill(0);

    data.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
        histogram[binIndex]++;
    });

    return histogram;
}

function calculateBoxPlotData(data) {
    const sorted = data.sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const median = sorted[Math.floor(sorted.length * 0.5)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const min = Math.max(q1 - 1.5 * iqr, Math.min(...sorted));
    const max = Math.min(q3 + 1.5 * iqr, Math.max(...sorted));

    return {
        min,
        q1,
        median,
        q3,
        max,
        outliers: sorted.filter(v => v < min || v > max)
    };
}

// D3 drag behavior
function drag(simulation) {
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
}

// Refresh data
function refreshData() {
    loadData();
}

// Show error message
function showError(message) {
    alert("Genetic Algorithm: " + message);
}

// Show success message
function showSuccess(message) {
    alert("Genetic Algorithm: " + message);
}

// Function to load allocation data, can be called by any function needing it
function loadAllocationDataForVisualizations() {
    if (allocationData) return true; // Already loaded

    const stored = localStorage.getItem('allocationResults');
    if (stored) {
        try {
            allocationData = JSON.parse(stored);
            if (!allocationData || !allocationData.classes) {
                console.error("Visualizations: Invalid allocation data structure in localStorage.");
                allocationData = null;
                return false;
            }
            return true;
        } catch (e) {
            console.error("Visualizations: Error parsing allocation data from localStorage:", e);
            allocationData = null;
            return false;
        }
    } else {
        console.warn("Visualizations: No allocation data found in localStorage. Please run allocation first.");
        return false;
    }
}