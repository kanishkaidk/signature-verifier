import { useEffect, useState } from 'react';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export default function AnalyticsDashboard({ history }) {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const checkDarkMode = () => {
      setIsDark(document.body.classList.contains('dark-mode'));
    };
    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  const textColor = isDark ? '#e5e5e5' : '#171717';
  const gridColor = isDark ? '#404040' : '#d4d4d4';

  const recentVerifications = history.filter(h => h.type === 'single').slice(-10);
  const batchVerifications = history.filter(h => h.type === 'batch');

  const genuineCount = recentVerifications.filter(h => h.result?.similarity_score >= 0.85).length;
  const forgeryCount = recentVerifications.length - genuineCount;

  const lineData = {
    labels: recentVerifications.map((_, i) => `V${i + 1}`),
    datasets: [
      {
        label: 'Similarity Score',
        data: recentVerifications.map(h => (h.result?.similarity_score || 0) * 100),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const doughnutData = {
    labels: ['Genuine', 'Forgery'],
    datasets: [
      {
        data: [genuineCount, forgeryCount],
        backgroundColor: ['#16a34a', '#dc2626'],
        borderWidth: 2,
        borderColor: isDark ? '#262626' : '#ffffff',
      },
    ],
  };

  const barData = {
    labels: ['< 50%', '50-70%', '70-85%', '85-95%', '> 95%'],
    datasets: [
      {
        label: 'Verifications',
        data: [
          recentVerifications.filter(h => (h.result?.similarity_score || 0) < 0.5).length,
          recentVerifications.filter(h => {
            const s = h.result?.similarity_score || 0;
            return s >= 0.5 && s < 0.7;
          }).length,
          recentVerifications.filter(h => {
            const s = h.result?.similarity_score || 0;
            return s >= 0.7 && s < 0.85;
          }).length,
          recentVerifications.filter(h => {
            const s = h.result?.similarity_score || 0;
            return s >= 0.85 && s < 0.95;
          }).length,
          recentVerifications.filter(h => (h.result?.similarity_score || 0) >= 0.95).length,
        ],
        backgroundColor: ['#dc2626', '#f59e0b', '#fbbf24', '#4ade80', '#16a34a'],
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: textColor },
      },
    },
    scales: {
      x: {
        ticks: { color: textColor },
        grid: { color: gridColor },
      },
      y: {
        ticks: { color: textColor },
        grid: { color: gridColor },
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: textColor },
      },
    },
  };

  if (recentVerifications.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <h3>No verification data yet</h3>
        <p>Start verifying signatures to see analytics</p>
      </div>
    );
  }

  return (
    <div className="analytics-grid">
      <div className="chart-card">
        <h3 className="chart-title">Recent Similarity Scores</h3>
        <div className="chart-container">
          <Line data={lineData} options={chartOptions} />
        </div>
      </div>

      <div className="chart-card">
        <h3 className="chart-title">Verification Results</h3>
        <div className="chart-container">
          <Doughnut data={doughnutData} options={doughnutOptions} />
        </div>
      </div>

      <div className="chart-card">
        <h3 className="chart-title">Score Distribution</h3>
        <div className="chart-container">
          <Bar data={barData} options={chartOptions} />
        </div>
      </div>

      <div className="chart-card">
        <h3 className="chart-title">Statistics</h3>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Total Verifications</div>
            <div className="stat-value">{history.length}</div>
            <div className="stat-sub">{recentVerifications.length} single, {batchVerifications.length} batch</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Avg. Score</div>
            <div className="stat-value">
              {recentVerifications.length > 0
                ? Math.round((recentVerifications.reduce((sum, h) => sum + (h.result?.similarity_score || 0), 0) / recentVerifications.length) * 100)
                : 0}%
            </div>
            <div className="stat-sub">Last 10 verifications</div>
          </div>
        </div>
      </div>
    </div>
  );
}
