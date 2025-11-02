import { useState, useEffect } from "react";
import { BarChart3, TrendingUp } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getHistory, getMetrics, type HistoryItem, type MetricsResponse } from "@/lib/api";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const Analytics = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const [historyData, metricsData] = await Promise.all([
          getHistory(),
          getMetrics(),
        ]);
        setHistory(historyData.history || []);
        setMetrics(metricsData);
      } catch (err: any) {
        setError(err.message || "Failed to load analytics");
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, []);

  const threshold = metrics?.threshold ?? 0.85;

  // Process data for charts
  const singleVerifications = history.filter((h) => h.type === "single");
  const batchVerifications = history.filter((h) => h.type === "batch");

  // Similarity scores over time (line chart)
  // Use combined_score if available (from detailed_metrics), fallback to similarity_score
  const similarityData = singleVerifications
    .slice(-10)
    .map((item, idx) => {
      const score = item.result?.detailed_metrics?.combined_score 
        || item.result?.similarity_score 
        || 0;
      return {
        name: `V${idx + 1}`,
        similarity: score * 100,
      };
    });

  // Verdict distribution (pie chart)
  // CRITICAL FIX: Use verdict from backend, NOT just score threshold
  // "Same person" = Genuine, "Different person" or contains "Forgery" = Forgery
  const genuineCount = singleVerifications.filter((h) => {
    if (!h.result) return false;
    
    // First check the actual verdict from backend
    const verdict = h.result.verdict?.toLowerCase() || "";
    if (verdict.includes("same person") || verdict.includes("match")) {
      return true;
    }
    if (verdict.includes("different person") || verdict.includes("forgery")) {
      return false;
    }
    
    // Fallback: Use combined_score if available, otherwise similarity_score
    const score = h.result.detailed_metrics?.combined_score 
      || h.result.similarity_score 
      || 0;
    
    // Use threshold, but be more lenient (0.70 instead of 0.85) for same person
    return score >= 0.70;
  }).length;
  
  const forgeryCount = singleVerifications.length - genuineCount;
  const verdictData = [
    { name: "Genuine", value: genuineCount, color: "#16a34a" },
    { name: "Forgery", value: forgeryCount, color: "#dc2626" },
  ];

  // Score distribution (bar chart)
  const scoreRanges = [
    { range: "< 50%", min: 0, max: 0.5 },
    { range: "50-70%", min: 0.5, max: 0.7 },
    { range: "70-85%", min: 0.7, max: 0.85 },
    { range: "85-95%", min: 0.85, max: 0.95 },
    { range: "> 95%", min: 0.95, max: 1.0 },
  ];

  const distributionData = scoreRanges.map((range) => ({
    range: range.range,
    count: singleVerifications.filter((h) => {
      const score = h.result?.detailed_metrics?.combined_score 
        || h.result?.similarity_score 
        || 0;
      return score >= range.min && score < range.max;
    }).length,
  }));

  // Average similarity - use combined_score if available
  const avgSimilarity =
    singleVerifications.length > 0
      ? singleVerifications.reduce(
          (sum, h) => {
            const score = h.result?.detailed_metrics?.combined_score 
              || h.result?.similarity_score 
              || 0;
            return sum + score;
          },
          0
        ) / singleVerifications.length
      : 0;

  if (isLoading) {
    return (
      <div className="min-h-screen pt-24 pb-16">
        <div className="container mx-auto px-6">
          <div className="text-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading analytics...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen pt-24 pb-16">
        <div className="container mx-auto px-6">
          <div className="text-center py-20">
            <p className="text-destructive mb-4">Error: {error}</p>
            <button onClick={() => window.location.reload()} className="px-4 py-2 bg-primary text-white rounded">
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (singleVerifications.length === 0) {
    return (
      <div className="min-h-screen pt-24 pb-16">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 space-y-4">
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
              Analytics Dashboard
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              View verification statistics and trends
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
              <BarChart3 className="w-16 h-16 text-muted-foreground" />
              <p className="text-lg font-medium">No verification data yet</p>
              <p className="text-muted-foreground">
                Start verifying signatures to see analytics
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 space-y-4">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
            Analytics Dashboard
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            View verification statistics and trends
          </p>
        </div>

        <div className="max-w-6xl mx-auto space-y-6">
          {/* Stats Cards */}
          <div className="grid md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Total Verifications</CardDescription>
                <CardTitle className="text-3xl">{history.length}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">
                  {singleVerifications.length} single, {batchVerifications.length} batch
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Average Score</CardDescription>
                <CardTitle className="text-3xl">{(avgSimilarity * 100).toFixed(1)}%</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">Last {singleVerifications.length} verifications</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Genuine</CardDescription>
                <CardTitle className="text-3xl text-green-600">{genuineCount}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">
                  {singleVerifications.length > 0
                    ? `${((genuineCount / singleVerifications.length) * 100).toFixed(1)}% of total`
                    : "0%"}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Forgery</CardDescription>
                <CardTitle className="text-3xl text-red-600">{forgeryCount}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">
                  {singleVerifications.length > 0
                    ? `${((forgeryCount / singleVerifications.length) * 100).toFixed(1)}% of total`
                    : "0%"}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Charts */}
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Recent Similarity Scores</CardTitle>
                <CardDescription>Last 10 verifications</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={similarityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="similarity"
                      stroke="#2563eb"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Verification Results</CardTitle>
                <CardDescription>Genuine vs Forgery</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={verdictData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {verdictData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Score Distribution</CardTitle>
                <CardDescription>Distribution of similarity scores</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={distributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
