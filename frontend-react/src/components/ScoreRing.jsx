import { useEffect, useState } from 'react';

export default function ScoreRing({ score, threshold = 0.85 }) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const scorePercent = Math.round(score * 100);
  const isMatch = score >= threshold;

  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference;

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(scorePercent);
    }, 100);
    return () => clearTimeout(timer);
  }, [scorePercent]);

  return (
    <div className="score-ring">
      <svg className="ring-svg" width="120" height="120" viewBox="0 0 120 120">
        <circle
          className="ring-bg"
          cx="60"
          cy="60"
          r={radius}
        />
        <circle
          className={`ring-progress ${isMatch ? 'success' : 'error'}`}
          cx="60"
          cy="60"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
        />
      </svg>
      <div className="ring-text">
        <span className="score">{scorePercent}%</span>
        <span className="label">Match</span>
      </div>
    </div>
  );
}
