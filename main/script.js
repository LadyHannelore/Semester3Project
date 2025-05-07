// Sample data for charts
const academicScores = {
    labels: ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'],
    datasets: [{
        label: 'Average Academic Score',
        data: [75, 82, 68, 90, 85],
        backgroundColor: 'rgba(66, 153, 225, 0.5)',
        borderColor: 'rgba(66, 153, 225, 1)',
        borderWidth: 1
    }]
};

const wellbeingScores = {
    labels: ['Class A', 'Class B', 'Class C', 'Class D', 'Class E'],
    datasets: [{
        label: 'Average Wellbeing Score',
        data: [80, 75, 85, 70, 90],
        borderColor: 'rgba(72, 187, 120, 1)',
        backgroundColor: 'rgba(72, 187, 120, 0.1)',
        tension: 0.4,
        fill: true
    }]
};

const bullyingRisk = {
    labels: ['Low Risk', 'Medium Risk', 'High Risk'],
    datasets: [{
        data: [45, 35, 20],
        backgroundColor: [
            'rgba(72, 187, 120, 0.8)',
            'rgba(246, 173, 85, 0.8)',
            'rgba(245, 101, 101, 0.8)'
        ],
        borderWidth: 1
    }]
};

// Chart configurations
const chartConfig = {
    type: 'bar',
    data: academicScores,
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Academic Scores by Class'
            }
        }
    }
};

const lineChartConfig = {
    type: 'line',
    data: wellbeingScores,
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Wellbeing Scores by Class'
            }
        }
    }
};

const pieChartConfig = {
    type: 'pie',
    data: bullyingRisk,
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Distribution of Bullying Risk Levels'
            }
        }
    }
};

// Initialize charts when the DOM is loaded
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

    // Create charts
    new Chart(
        document.getElementById('academicScoreChart'),
        chartConfig
    );

    new Chart(
        document.getElementById('wellbeingScoreChart'),
        lineChartConfig
    );

    new Chart(
        document.getElementById('bullyingRiskChart'),
        pieChartConfig
    );

    // Populate recent activity log
    populateActivityLog();
});

// Sample activity log data
const recentActivities = [
    {
        datetime: '2024-03-15 14:30',
        modelName: 'KMeans+CP-SAT',
        result: 'Successfully allocated 95% of students'
    },
    {
        datetime: '2024-03-15 13:15',
        modelName: 'Genetic Algorithm',
        result: 'Optimized for academic balance'
    },
    {
        datetime: '2024-03-15 11:45',
        modelName: 'KMeans+CP-SAT',
        result: 'Reduced bullying risk by 15%'
    },
    {
        datetime: '2024-03-15 10:20',
        modelName: 'Random Forest',
        result: 'Identified 25 high-risk students'
    },
    {
        datetime: '2024-03-15 09:00',
        modelName: 'KMeans+CP-SAT',
        result: 'Initial allocation completed'
    }
];

// Function to populate activity log
function populateActivityLog() {
    const activityLog = document.getElementById('activityLog');
    recentActivities.forEach(activity => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${activity.datetime}</td>
            <td>${activity.modelName}</td>
            <td>${activity.result}</td>
        `;
        activityLog.appendChild(row);
    });
}

// Navigation function
function navigateTo(page) {
    // This would be replaced with actual navigation logic
    console.log(`Navigating to ${page} page`);
    // For now, just show an alert
    alert(`This would navigate to the ${page} page`);
} 