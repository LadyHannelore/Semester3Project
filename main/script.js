// Initialize dashboard when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    initializeNavigation();
});

// Initialize navigation links and prevent default behavior for empty links
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href && href !== '#') {
                // Allow default navigation for actual links
            } else {
                e.preventDefault();
            }
        });
    });
}

// Load dashboard data from localStorage and update UI components
function loadDashboardData() {
    const studentData = JSON.parse(localStorage.getItem('classforgeDataset'));
    const allocationData = JSON.parse(localStorage.getItem('allocationResults'));
    const modelComparisonData = JSON.parse(localStorage.getItem('modelResults'));

    updateSummaryCards(studentData, allocationData);
    initializeCharts(studentData);
    updateRecentActivity(modelComparisonData, studentData);
}

// Update summary cards with student and allocation data
function updateSummaryCards(studentData, allocationData) {
    let totalStudents = 0;
    let avgAcademicScore = 0;
    let avgWellbeingScore = 0;
    let highRiskBulliesCount = 0;
    let academicScores = [];
    let wellbeingScores = [];

    if (studentData && studentData.rows && studentData.rows.length > 0) {
        totalStudents = studentData.rows.length;
        const academicIndex = studentData.headers.indexOf('Academic_Performance');
        const wellbeingIndex = studentData.headers.indexOf('Wellbeing_Score');
        const bullyingIndex = studentData.headers.indexOf('Bullying_Score');

        studentData.rows.forEach(row => {
            if (academicIndex !== -1) {
                const score = parseFloat(row[academicIndex]);
                if (!isNaN(score)) academicScores.push(score);
            }
            if (wellbeingIndex !== -1) {
                const score = parseFloat(row[wellbeingIndex]);
                if (!isNaN(score)) wellbeingScores.push(score);
            }
            if (bullyingIndex !== -1) {
                const score = parseFloat(row[bullyingIndex]);
                if (!isNaN(score) && score >= 6) highRiskBulliesCount++;
            }
        });

        if (academicScores.length > 0) {
            avgAcademicScore = academicScores.reduce((sum, score) => sum + score, 0) / academicScores.length;
        }
        if (wellbeingScores.length > 0) {
            avgWellbeingScore = wellbeingScores.reduce((sum, score) => sum + score, 0) / wellbeingScores.length;
        }
    }

    document.getElementById('totalStudents').textContent = totalStudents;
    document.getElementById('avgAcademicScore').textContent = avgAcademicScore.toFixed(1);
    document.getElementById('avgWellbeingScore').textContent = avgWellbeingScore.toFixed(1);
    document.getElementById('highRiskBullies').textContent = totalStudents > 0 ? ((highRiskBulliesCount / totalStudents) * 100).toFixed(1) + '%' : '0%';

    let totalClasses = 0;
    let unassignedStudents = 0;
    if (allocationData && allocationData.classes) {
        totalClasses = allocationData.classes.length;
        const assignedStudentIds = new Set();
        allocationData.classes.forEach(cls => {
            cls.students.forEach(s => assignedStudentIds.add(String(s.id)));
        });
        if (studentData && studentData.rows) {
            unassignedStudents = studentData.rows.filter(row => {
                const studentIdIndex = studentData.headers.indexOf('Student_ID');
                return studentIdIndex !== -1 && !assignedStudentIds.has(String(row[studentIdIndex]));
            }).length;
        }
    } else if (studentData && studentData.rows) {
        unassignedStudents = totalStudents;
    }

    document.getElementById('totalClasses').textContent = totalClasses;
    document.getElementById('unassignedStudents').textContent = unassignedStudents;
}

// Initialize charts with student data
function initializeCharts(studentData) {
    // Remove loading indicators
    ['academicScoreChartLoading', 'wellbeingScoreChartLoading', 'bullyingRiskChartLoading'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });

    const defaultChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
        },
        scales: {
            y: { beginAtZero: true, ticks: { precision: 0 } },
            x: { grid: { display: false } }
        }
    };

    // Academic Score Chart (Histogram)
    if (document.getElementById('academicScoreChart')) {
        const academicScores = studentData && studentData.rows ? studentData.rows.map(row => parseFloat(row[studentData.headers.indexOf('Academic_Performance')])).filter(s => !isNaN(s)) : [];
        const academicBins = createHistogramBins(academicScores, 0, 100, 10);
        new Chart(document.getElementById('academicScoreChart'), {
            type: 'bar',
            data: {
                labels: academicBins.map(b => `${b.min}-${b.max}`),
                datasets: [{
                    label: 'Academic Score',
                    data: academicBins.map(b => b.count),
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {...defaultChartOptions, plugins: {...defaultChartOptions.plugins, title: { display: false }}}
        });
    }

    // Wellbeing Score Chart (Histogram)
    if (document.getElementById('wellbeingScoreChart')) {
        const wellbeingScores = studentData && studentData.rows ? studentData.rows.map(row => parseFloat(row[studentData.headers.indexOf('Wellbeing_Score')])).filter(s => !isNaN(s)) : [];
        const wellbeingBins = createHistogramBins(wellbeingScores, 0, 10, 5);
        new Chart(document.getElementById('wellbeingScoreChart'), {
            type: 'bar',
            data: {
                labels: wellbeingBins.map(b => `${b.min}-${b.max}`),
                datasets: [{
                    label: 'Wellbeing Score',
                    data: wellbeingBins.map(b => b.count),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {...defaultChartOptions, plugins: {...defaultChartOptions.plugins, title: { display: false }}}
        });
    }

    // Bullying Risk Chart (Pie)
    if (document.getElementById('bullyingRiskChart')) {
        const bullyingScores = studentData && studentData.rows ? studentData.rows.map(row => parseFloat(row[studentData.headers.indexOf('Bullying_Score')])).filter(s => !isNaN(s)) : [];
        const riskLevels = { 'Low (0-2)': 0, 'Medium (3-5)': 0, 'High (6-10)': 0 };
        bullyingScores.forEach(score => {
            if (score <= 2) riskLevels['Low (0-2)']++;
            else if (score <= 5) riskLevels['Medium (3-5)']++;
            else riskLevels['High (6-10)']++;
        });
        new Chart(document.getElementById('bullyingRiskChart'), {
            type: 'pie',
            data: {
                labels: Object.keys(riskLevels),
                datasets: [{
                    label: 'Bullying Risk',
                    data: Object.values(riskLevels),
                    backgroundColor: [
                        'rgba(72, 187, 120, 0.8)',
                        'rgba(246, 173, 85, 0.8)',
                        'rgba(245, 101, 101, 0.8)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top' },
                    title: { display: false }
                }
            }
        });
    }

    // Limit class-based charts to first 10 classes if such charts exist
    // Example: If you have a chart showing per-class data, slice to 10 classes:
    // const classLabels = allClassLabels.slice(0, 10);
    // const classData = allClassData.slice(0, 10);
    // ... use classLabels and classData in your chart config ...
}

// Helper: Histogram binning
function createHistogramBins(data, min, max, bins) {
    if (!data || data.length === 0) {
        return Array.from({length: bins}, (_, i) => ({
            min: min + i * ((max - min) / bins),
            max: min + (i + 1) * ((max - min) / bins),
            count: 0
        }));
    }
    const binSize = (max - min) / bins;
    const result = Array.from({length: bins}, (_, i) => ({
        min: min + i * binSize,
        max: min + (i + 1) * binSize,
        count: 0
    }));
    data.forEach(val => {
        let idx = Math.floor((val - min) / binSize);
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        result[idx].count++;
    });
    return result;
}

// Update recent activity log with model comparison and allocation data
function updateRecentActivity(modelComparisonData, studentData) {
    const activityLog = document.getElementById('activityLog');
    activityLog.innerHTML = '';

    // Example: Compose recent activity from localStorage events
    const activities = [];

    // Model comparison activity
    if (modelComparisonData && Array.isArray(modelComparisonData) && modelComparisonData.length > 0) {
        const preferred = modelComparisonData.find(m => m.preferred);
        if (preferred) {
            activities.push({
                time: new Date().toLocaleString(),
                activity: 'Model Selected',
                details: `Preferred model: ${preferred.modelName}`
            });
        }
    }

    // Student data upload/generation
    if (studentData && studentData.source) {
        activities.push({
            time: new Date().toLocaleString(),
            activity: studentData.source === 'upload' ? 'Data Uploaded' : 'Synthetic Data Generated',
            details: `Rows: ${studentData.rows.length}`
        });
    }

    // Allocation run
    const allocationData = JSON.parse(localStorage.getItem('allocationResults'));
    if (allocationData && allocationData.metrics) {
        activities.push({
            time: new Date().toLocaleString(),
            activity: 'Group Allocation',
            details: `Classes: ${allocationData.metrics.numClasses}, Students: ${allocationData.metrics.totalStudents}`
        });
    }

    // Fallback if no activity
    if (activities.length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="3" style="text-align:center;color:#a0aec0;">No recent activity found.</td>`;
        activityLog.appendChild(tr);
        return;
    }

    // Show up to 10 recent activities
    activities.slice(0, 10).forEach(act => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${act.time}</td>
            <td>${act.activity}</td>
            <td>${act.details}</td>
        `;
        activityLog.appendChild(tr);
    });
}