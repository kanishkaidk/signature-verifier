import { useState, useEffect } from "react";
import { ScrollText, FileText, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getHistory, getMetrics, type HistoryItem, type MetricsResponse } from "@/lib/api";
import { format } from "date-fns";

const History = () => {
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
        setError(err.message || "Failed to load history");
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, []);

  const threshold = metrics?.threshold ?? 0.85;

  if (isLoading) {
    return (
      <div className="min-h-screen pt-24 pb-16">
        <div className="container mx-auto px-6">
          <div className="text-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading history...</p>
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
            <Button onClick={() => window.location.reload()}>Retry</Button>
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
            Verification History
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Review past verifications and results
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
              <ScrollText className="w-16 h-16 text-muted-foreground" />
              <p className="text-lg font-medium">No history yet</p>
              <p className="text-muted-foreground">
                Your verification history will appear here
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {history
                .slice()
                .reverse()
                .map((item, index) => (
                  <Card key={index}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="text-lg">
                            {item.type === "single" ? "Single Verification" : "Batch Verification"}
                          </CardTitle>
                          <CardDescription>
                            {format(new Date(item.timestamp), "PPpp")}
                          </CardDescription>
                        </div>
                        {item.type === "single" && item.result && (
                          <Badge
                            variant={
                              item.result.similarity_score >= threshold
                                ? "default"
                                : "destructive"
                            }
                          >
                            {item.result.verdict || 
                              (item.result.similarity_score >= threshold ? "Match" : "Forgery")}
                          </Badge>
                        )}
                        {item.type === "batch" && (
                          <Badge variant="outline">
                            {item.count || 0} files
                          </Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent>
                      {item.type === "single" && item.result && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Similarity Score:</span>
                            <span className="font-semibold">
                              {(item.result.similarity_score * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-muted-foreground">Verdict:</span>
                            <span className="font-semibold">{item.result.verdict}</span>
                          </div>
                        </div>
                      )}
                      {item.type === "batch" && item.results && (
                        <div className="space-y-2">
                          <p className="text-sm text-muted-foreground mb-2">
                            Top {item.results.length} results:
                          </p>
                          <div className="space-y-1">
                            {item.results.slice(0, 5).map((result, idx) => (
                              <div
                                key={idx}
                                className="flex justify-between items-center text-sm"
                              >
                                <span className="text-muted-foreground">
                                  {result.filename || `File ${idx + 1}`}
                                </span>
                                {result.similarity_score !== undefined ? (
                                  <>
                                    <span className="font-medium">
                                      {(result.similarity_score * 100).toFixed(1)}%
                                    </span>
                                    <Badge
                                      variant={
                                        result.similarity_score >= threshold
                                          ? "default"
                                          : "destructive"
                                      }
                                      className="ml-2"
                                    >
                                      {result.verdict || 
                                        (result.similarity_score >= threshold ? "Match" : "Forgery")}
                                    </Badge>
                                  </>
                                ) : (
                                  <Badge variant="destructive">Error</Badge>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default History;
