import { useState, useEffect } from "react";
import { Star, FolderOpen, Play, FileImage } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { batchVerify, getMetrics, type BatchResponse, type MetricsResponse } from "@/lib/api";

const Batch = () => {
  const [reference, setReference] = useState<File | null>(null);
  const [referenceUrl, setReferenceUrl] = useState<string>("");
  const [comparisons, setComparisons] = useState<FileList | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch(() => {});
  }, []);

  const handleReferenceSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setReference(file);
      setReferenceUrl(URL.createObjectURL(file));
    }
  };

  const handleReferenceDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setReference(file);
      setReferenceUrl(URL.createObjectURL(file));
    }
  };

  const handleComparisonsSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) setComparisons(files);
  };

  const handleComparisonsDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files) setComparisons(files);
  };

  const handleBatchVerify = async () => {
    if (!reference || !comparisons || comparisons.length === 0) {
      toast({
        title: "Missing files",
        description: "Please upload reference and comparison signatures",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setResult(null);

    try {
      const response = await batchVerify(reference, comparisons);
      setResult(response);

      toast({
        title: "Batch verification complete",
        description: `Processed ${response.results.length} signatures`,
      });
    } catch (error: any) {
      toast({
        title: "Batch verification failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const threshold = metrics?.threshold ?? 0.85;

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 space-y-4">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
            Batch Verification
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Upload one reference signature and multiple files to compare against it
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="space-y-6 mb-8">
            {/* Reference Signature */}
            <div
              onDrop={handleReferenceDrop}
              onDragOver={(e) => e.preventDefault()}
              className="relative border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-accent transition-colors cursor-pointer bg-card"
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleReferenceSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Star className="w-12 h-12 mx-auto mb-4 text-accent" />
              <p className="font-medium mb-2">Reference Signature</p>
              <p className="text-sm text-muted-foreground">
                {reference ? reference.name : "No file chosen"}
              </p>
            </div>

            {/* Reference Preview */}
            {referenceUrl && (
              <div className="border rounded-lg p-4 bg-card flex items-center justify-center">
                <img src={referenceUrl} alt="Reference" className="max-w-full max-h-48 object-contain" />
              </div>
            )}

            {/* Comparison Signatures */}
            <div
              onDrop={handleComparisonsDrop}
              onDragOver={(e) => e.preventDefault()}
              className="relative border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer bg-card"
            >
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={handleComparisonsSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <FolderOpen className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="font-medium mb-2">Comparison Signatures</p>
              <p className="text-sm text-muted-foreground">
                {comparisons ? `${comparisons.length} files selected` : "No files chosen"}
              </p>
            </div>

            <Button
              onClick={handleBatchVerify}
              disabled={!reference || !comparisons || isProcessing}
              className="w-full"
              size="lg"
            >
              <Play className="w-4 h-4 mr-2" />
              {isProcessing ? "Processing..." : "Run Batch Verification"}
            </Button>
          </div>

          {/* Loading */}
          {isProcessing && (
            <div className="space-y-4 p-6 rounded-lg bg-card border border-border mb-8">
              <p className="text-sm text-muted-foreground">Processing signatures...</p>
              <Progress value={66} />
            </div>
          )}

          {/* Results Table */}
          {result && !isProcessing && (
            <div className="border rounded-lg overflow-hidden bg-card">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>File</TableHead>
                    <TableHead className="text-right">Similarity</TableHead>
                    <TableHead className="text-right">Verdict</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.results.map((item, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-medium">{item.filename || `File ${index + 1}`}</TableCell>
                      <TableCell className="text-right">
                        {item.error ? (
                          <span className="text-muted-foreground">Error</span>
                        ) : (
                          <strong>{(item.similarity_score! * 100).toFixed(1)}%</strong>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        {item.error ? (
                          <Badge variant="destructive">{item.error}</Badge>
                        ) : (
                          <Badge
                            variant={
                              item.similarity_score! >= threshold ? "default" : "destructive"
                            }
                          >
                            {item.verdict || (item.similarity_score! >= threshold ? "Match" : "Forgery")}
                          </Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}

          {/* Empty State */}
          {!result && !isProcessing && (
            <div className="text-center py-12 text-muted-foreground">
              <FileImage className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Results will appear here after verification</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Batch;
