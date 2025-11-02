import { useState, useEffect } from "react";
import { AlertCircle, CheckCircle2, FileImage, Flame, Download, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import {
  verifySignatures,
  generateSaliencyHeatmap,
  generateDualSaliency,
  generateDifferenceHeatmap,
  generateSaliencyDifference,
  generateGradCamHeatmap,
  generateStrokeOverlay,
  getPreprocessedPreview,
  getNormalizedOverlay,
  generateAlignPreview,
  detectSignaturesInDocument,
  downloadReport,
  getMetrics,
  getDisclaimer,
  healthCheck,
  API_BASE_URL,
  type PredictResponse,
  type MetricsResponse,
} from "@/lib/api";

const Verify = () => {
  const [signature1, setSignature1] = useState<File | null>(null);
  const [signature2, setSignature2] = useState<File | null>(null);
  const [signature1Url, setSignature1Url] = useState<string>("");
  const [signature2Url, setSignature2Url] = useState<string>("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isGeneratingHeatmap, setIsGeneratingHeatmap] = useState(false);
  const [isGeneratingGradCam, setIsGeneratingGradCam] = useState(false);
  const [isGeneratingDualSaliency, setIsGeneratingDualSaliency] = useState(false);
  const [isGeneratingDifference, setIsGeneratingDifference] = useState(false);
  const [isGeneratingSaliencyDiff, setIsGeneratingSaliencyDiff] = useState(false);
  const [isDownloadingReport, setIsDownloadingReport] = useState(false);
  const [heatmapUrl, setHeatmapUrl] = useState<string>("");
  const [gradcamUrl, setGradcamUrl] = useState<string>("");
  const [dualSaliencyUrl, setDualSaliencyUrl] = useState<string>("");
  const [differenceUrl, setDifferenceUrl] = useState<string>("");
  const [saliencyDiffUrl, setSaliencyDiffUrl] = useState<string>("");
  const [heatmapType, setHeatmapType] = useState<"saliency" | "gradcam" | "dual_saliency" | "difference" | "saliency_diff" | "stroke_overlay" | "preprocessed" | "normalized_overlay" | "align_preview" | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [strokeOverlayUrl, setStrokeOverlayUrl] = useState<string | null>(null);
  const [preprocessedUrl, setPreprocessedUrl] = useState<string | null>(null);
  const [normalizedOverlayUrl, setNormalizedOverlayUrl] = useState<string | null>(null);
  const [alignPreviewUrl, setAlignPreviewUrl] = useState<string | null>(null);
  const [isGeneratingStrokeOverlay, setIsGeneratingStrokeOverlay] = useState(false);
  const [isGeneratingPreprocessed, setIsGeneratingPreprocessed] = useState(false);
  const [isGeneratingNormalizedOverlay, setIsGeneratingNormalizedOverlay] = useState(false);
  const [isGeneratingAlignPreview, setIsGeneratingAlignPreview] = useState(false);
  const [detectedSignatures, setDetectedSignatures] = useState<any[] | null>(null);
  const [isDetectingSignatures, setIsDetectingSignatures] = useState(false);
  const [overlay, setOverlay] = useState(50);
  const [zoom, setZoom] = useState(100);
  const [disclaimer, setDisclaimer] = useState<string>("");
  const { toast } = useToast();

  // Load metrics and disclaimer on mount + test connection
  useEffect(() => {
    // Test backend connection first
    healthCheck()
      .then(() => {
        // Connection OK, load data
        return Promise.all([
          getMetrics().then(setMetrics).catch(() => {
            toast({
              title: "Warning",
              description: "Could not load model metrics",
              variant: "destructive",
            });
          }),
          getDisclaimer()
            .then(data => setDisclaimer(data.disclaimer))
            .catch(() => {})
        ]);
      })
      .catch((error: any) => {
        toast({
          title: "⚠️ Backend Connection Failed",
          description: `Cannot connect to ${API_BASE_URL}. Make sure Flask server is running on port 5000. Error: ${error.message}`,
          variant: "destructive",
        });
      });
  }, [toast]);

  // Function to refresh current visualization with new opacity
  const handleRefreshVisualization = async () => {
    if (!signature1 || !signature2 || !heatmapType) {
      return;
    }

    const opacity = overlay / 100;
    let refreshFn: () => Promise<void>;

    switch (heatmapType) {
      case "saliency":
        refreshFn = async () => {
          setIsGeneratingHeatmap(true);
          try {
            const blob = await generateSaliencyHeatmap(signature1, signature2, opacity);
            const url = URL.createObjectURL(blob);
            setHeatmapUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingHeatmap(false);
          }
        };
        break;
      case "gradcam":
        refreshFn = async () => {
          setIsGeneratingGradCam(true);
          try {
            const blob = await generateGradCamHeatmap(signature1, signature2, opacity);
            const url = URL.createObjectURL(blob);
            setGradcamUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingGradCam(false);
          }
        };
        break;
      case "dual_saliency":
        refreshFn = async () => {
          setIsGeneratingDualSaliency(true);
          try {
            const blob = await generateDualSaliency(signature1, signature2, opacity);
            const url = URL.createObjectURL(blob);
            setDualSaliencyUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingDualSaliency(false);
          }
        };
        break;
      case "difference":
        refreshFn = async () => {
          setIsGeneratingDifference(true);
          try {
            const blob = await generateDifferenceHeatmap(signature1, signature2, opacity);
            const url = URL.createObjectURL(blob);
            setDifferenceUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingDifference(false);
          }
        };
        break;
      case "saliency_diff":
        refreshFn = async () => {
          setIsGeneratingSaliencyDiff(true);
          try {
            const blob = await generateSaliencyDifference(signature1, signature2, opacity);
            const url = URL.createObjectURL(blob);
            setSaliencyDiffUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingSaliencyDiff(false);
          }
        };
        break;
      case "normalized_overlay":
        refreshFn = async () => {
          setIsGeneratingNormalizedOverlay(true);
          try {
            const blob = await getNormalizedOverlay(signature1, signature2, { opacity });
            const url = URL.createObjectURL(blob);
            setNormalizedOverlayUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingNormalizedOverlay(false);
          }
        };
        break;
      case "align_preview":
        refreshFn = async () => {
          setIsGeneratingAlignPreview(true);
          try {
            const blob = await generateAlignPreview(signature1, signature2, { opacity });
            const url = URL.createObjectURL(blob);
            setAlignPreviewUrl(url);
          } catch (error: any) {
            toast({
              title: "Refresh failed",
              description: error.message || "Could not refresh visualization",
              variant: "destructive",
            });
          } finally {
            setIsGeneratingAlignPreview(false);
          }
        };
        break;
      default:
        return;
    }

    await refreshFn();
  };

  const handleDrop = (e: React.DragEvent, setter: (file: File) => void, urlSetter: (url: string) => void) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setter(file);
      urlSetter(URL.createObjectURL(file));
    } else {
      toast({
        title: "Invalid file",
        description: "Please upload an image file",
        variant: "destructive",
      });
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, setter: (file: File) => void, urlSetter: (url: string) => void) => {
    const file = e.target.files?.[0];
    if (file) {
      setter(file);
      urlSetter(URL.createObjectURL(file));
    }
  };

  const handleVerify = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures to verify",
        variant: "destructive",
      });
      return;
    }

    setIsVerifying(true);
    setResult(null);
    setHeatmapUrl("");
    setGradcamUrl("");
    setDualSaliencyUrl("");
    setDifferenceUrl("");
    setSaliencyDiffUrl("");
    setStrokeOverlayUrl(null);
    setPreprocessedUrl(null);
    setNormalizedOverlayUrl(null);
    setAlignPreviewUrl(null);
    setHeatmapType(null);
    setShowHeatmap(false);

    try {
      const response = await verifySignatures(signature1, signature2);
      setResult(response);

      // AUTO-SHOW PREPROCESSING PREVIEW after verification
      // This shows the user the alignment pipeline in action
      try {
        const previewBlob = await getPreprocessedPreview(signature1, signature2);
        const previewUrl = URL.createObjectURL(previewBlob);
        setPreprocessedUrl(previewUrl);
        setHeatmapType("preprocessed");
        setShowHeatmap(true);
        
        toast({
          title: "Verification complete",
          description: `Similarity: ${(response.similarity_score * 100).toFixed(1)}% - ${response.verdict}. Preprocessing preview auto-displayed.`,
        });
      } catch (previewError) {
        // If preview fails, still show verification result
        toast({
          title: "Verification complete",
          description: `Similarity: ${(response.similarity_score * 100).toFixed(1)}% - ${response.verdict}`,
        });
      }
    } catch (error: any) {
      toast({
        title: "Verification failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsVerifying(false);
    }
  };

  const handleGenerateHeatmap = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingHeatmap(true);
    try {
      const opacity = overlay / 100; // Convert percentage to 0-1
      const blob = await generateSaliencyHeatmap(signature1, signature2, opacity);
      const url = URL.createObjectURL(blob);
      setHeatmapUrl(url);
      setHeatmapType("saliency");
      setShowHeatmap(true);

      toast({
        title: "Saliency heatmap generated",
        description: "AI attention visualization is now available",
      });
    } catch (error: any) {
      toast({
        title: "Heatmap generation failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingHeatmap(false);
    }
  };

  const handleGenerateGradCam = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingGradCam(true);
    try {
      const opacity = overlay / 100; // Convert percentage to 0-1
      const blob = await generateGradCamHeatmap(signature1, signature2, opacity);
      const url = URL.createObjectURL(blob);
      setGradcamUrl(url);
      setHeatmapType("gradcam");
      setShowHeatmap(true);

      toast({
        title: "Grad-CAM heatmap generated",
        description: "Deep learning attention visualization is now available",
      });
    } catch (error: any) {
      toast({
        title: "Grad-CAM generation failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingGradCam(false);
    }
  };

  const handleGenerateDualSaliency = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingDualSaliency(true);
    try {
      const opacity = overlay / 100; // Convert percentage to 0-1
      const blob = await generateDualSaliency(signature1, signature2, opacity);
      const url = URL.createObjectURL(blob);
      setDualSaliencyUrl(url);
      setHeatmapType("dual_saliency");
      setShowHeatmap(true);

      toast({
        title: "Dual saliency maps generated",
        description: "Side-by-side saliency visualization available",
      });
    } catch (error: any) {
      toast({
        title: "Dual saliency generation failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingDualSaliency(false);
    }
  };

  const handleGenerateDifference = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingDifference(true);
    try {
      const opacity = overlay / 100; // Convert percentage to 0-1
      const blob = await generateDifferenceHeatmap(signature1, signature2, opacity);
      const url = URL.createObjectURL(blob);
      setDifferenceUrl(url);
      setHeatmapType("difference");
      setShowHeatmap(true);

      toast({
        title: "Difference heatmap generated",
        description: "Embedding difference visualization available",
      });
    } catch (error: any) {
      toast({
        title: "Difference heatmap generation failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingDifference(false);
    }
  };

  const handleGenerateSaliencyDiff = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingSaliencyDiff(true);
    try {
      const opacity = overlay / 100; // Convert percentage to 0-1
      const blob = await generateSaliencyDifference(signature1, signature2, opacity);
      const url = URL.createObjectURL(blob);
      setSaliencyDiffUrl(url);
      setHeatmapType("saliency_diff");
      setShowHeatmap(true);

      toast({
        title: "Saliency difference generated",
        description: "Highlighting differences in attention patterns",
      });
    } catch (error: any) {
      toast({
        title: "Saliency difference generation failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingSaliencyDiff(false);
    }
  };

  const handleGenerateStrokeOverlay = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingStrokeOverlay(true);
    try {
      const opacity = overlay / 100;
      const blob = await generateStrokeOverlay(signature1, signature2, opacity);
      const url = URL.createObjectURL(blob);
      setStrokeOverlayUrl(url);
      setHeatmapType("stroke_overlay");
      setShowHeatmap(true);

      toast({
        title: "Stroke overlay generated",
        description: "Shows exact stroke alignment - Red: unique to sig1, Green: unique to sig2, Yellow: matched strokes",
      });
    } catch (error: any) {
      toast({
        title: "Stroke overlay generation failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingStrokeOverlay(false);
    }
  };

  const handleGeneratePreprocessed = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingPreprocessed(true);
    try {
      const blob = await getPreprocessedPreview(signature1, signature2);
      const url = URL.createObjectURL(blob);
      setPreprocessedUrl(url);
      setHeatmapType("preprocessed");
      setShowHeatmap(true);

      toast({
        title: "Preprocessing preview generated",
        description: "Shows noise removal, alignment, and signature extraction steps",
      });
    } catch (error: any) {
      toast({
        title: "Preprocessing preview failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingPreprocessed(false);
    }
  };

  const handleGenerateNormalizedOverlay = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingNormalizedOverlay(true);
    try {
      const opacity = overlay / 100;
      const blob = await getNormalizedOverlay(signature1, signature2, {
        show_baseline: true,
        enable_baseline_align: true,
        enable_brightness_match: true,
        opacity,
      });
      const url = URL.createObjectURL(blob);
      setNormalizedOverlayUrl(url);
      setHeatmapType("normalized_overlay");
      setShowHeatmap(true);

      toast({
        title: "Normalized overlay generated",
        description: "Perfect alignment with baseline markers",
      });
    } catch (error: any) {
      toast({
        title: "Normalized overlay failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingNormalizedOverlay(false);
    }
  };

  const handleGenerateAlignPreview = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsGeneratingAlignPreview(true);
    try {
      const opacity = overlay / 100;
      const blob = await generateAlignPreview(signature1, signature2, {
        use_advanced: true,
        opacity,
      });
      const url = URL.createObjectURL(blob);
      setAlignPreviewUrl(url);
      setHeatmapType("align_preview");
      setShowHeatmap(true);
      
      toast({
        title: "Alignment preview generated",
        description: "Shows exact overlap visualization",
      });
    } catch (error: any) {
      toast({
        title: "Alignment preview failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingAlignPreview(false);
    }
  };

  const handleDetectSignatures = async (imageFile: File) => {
    setIsDetectingSignatures(true);
    try {
      const result = await detectSignaturesInDocument(imageFile);
      setDetectedSignatures(result.signatures);
      toast({
        title: "Signatures detected",
        description: `Found ${result.signatures_found} signature(s) in document`,
      });
    } catch (error: any) {
      toast({
        title: "Signature detection failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsDetectingSignatures(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!signature1 || !signature2) {
      toast({
        title: "Missing signatures",
        description: "Please upload both signatures first",
        variant: "destructive",
      });
      return;
    }

    setIsDownloadingReport(true);
    try {
      let heatmapBlob: Blob | undefined;
      if (heatmapType === "saliency" && heatmapUrl) {
        const response = await fetch(heatmapUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "gradcam" && gradcamUrl) {
        const response = await fetch(gradcamUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "dual_saliency" && dualSaliencyUrl) {
        const response = await fetch(dualSaliencyUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "difference" && differenceUrl) {
        const response = await fetch(differenceUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "saliency_diff" && saliencyDiffUrl) {
        const response = await fetch(saliencyDiffUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "stroke_overlay" && strokeOverlayUrl) {
        const response = await fetch(strokeOverlayUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "preprocessed" && preprocessedUrl) {
        const response = await fetch(preprocessedUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "normalized_overlay" && normalizedOverlayUrl) {
        const response = await fetch(normalizedOverlayUrl);
        heatmapBlob = await response.blob();
      } else if (heatmapType === "align_preview" && alignPreviewUrl) {
        const response = await fetch(alignPreviewUrl);
        heatmapBlob = await response.blob();
      }

      const pdfBlob = await downloadReport(signature1, signature2, heatmapBlob);
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `signguard_report_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      toast({
        title: "Report downloaded",
        description: "PDF report has been downloaded",
      });
    } catch (error: any) {
      toast({
        title: "Report download failed",
        description: error.message || "An error occurred",
        variant: "destructive",
      });
    } finally {
      setIsDownloadingReport(false);
    }
  };

  const threshold = metrics?.threshold ?? 0.92; // Updated to match stricter backend threshold
  const getVerdictColor = () => {
    if (!result) return "text-muted-foreground";
    const score = result.similarity_score;
    if (score >= 0.92) return "text-green-600 dark:text-green-400";
    if (score >= 0.85) return "text-yellow-600 dark:text-yellow-400"; // Uncertain zone
    return "text-red-600 dark:text-red-400";
  };

  const getVerdictText = () => {
    if (!result) return null;
    // Use verdict from backend which includes "Uncertain" state
    return result.verdict || (result.similarity_score >= 0.92 ? "Match" : result.similarity_score >= 0.85 ? "Uncertain" : "Forgery Detected");
  };

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="container mx-auto px-6">
        {/* Hero Section */}
        <div className="text-center mb-16 space-y-4">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
            AI-Powered Signature Verification
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Detect forgeries with deep learning precision using advanced Siamese networks
          </p>
        </div>

        {/* Verification Interface */}
        <div className="max-w-5xl mx-auto">
          <div className="mb-8">
            <h2 className="text-2xl font-semibold mb-2">Signature Verification</h2>
            <p className="text-muted-foreground">Upload two signature images to compare and verify authenticity</p>
          </div>

          {/* Upload Zones */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <div
              onDrop={(e) => handleDrop(e, setSignature1, setSignature1Url)}
              onDragOver={(e) => e.preventDefault()}
              className="relative border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer bg-card"
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFileSelect(e, setSignature1, setSignature1Url)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <FileImage className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="font-medium mb-2">Drop Signature 1 here</p>
              <p className="text-sm text-muted-foreground">
                {signature1 ? signature1.name : "No file chosen"}
              </p>
            </div>

            <div
              onDrop={(e) => handleDrop(e, setSignature2, setSignature2Url)}
              onDragOver={(e) => e.preventDefault()}
              className="relative border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer bg-card"
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleFileSelect(e, setSignature2, setSignature2Url)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <FileImage className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="font-medium mb-2">Drop Signature 2 here</p>
              <p className="text-sm text-muted-foreground">
                {signature2 ? signature2.name : "No file chosen"}
              </p>
            </div>
          </div>

          {/* Image Previews */}
          {(signature1Url || signature2Url) && (
            <div className="grid md:grid-cols-3 gap-4 mb-8">
              {signature1Url && (
                <div className="border rounded-lg p-4 bg-card flex items-center justify-center overflow-hidden">
                  <img
                    src={signature1Url}
                    alt="Signature 1"
                    className="max-w-full max-h-64 object-contain"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                </div>
              )}
              
              <div className="border rounded-lg p-4 bg-card flex items-center justify-center overflow-hidden relative">
                {signature1Url && (
                  <img
                    src={signature1Url}
                    alt="Base"
                    className="max-w-full max-h-64 object-contain absolute"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                )}
                {showHeatmap && heatmapType === "saliency" && heatmapUrl ? (
                  <img
                    src={heatmapUrl}
                    alt="Saliency Heatmap"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "gradcam" && gradcamUrl ? (
                  <img
                    src={gradcamUrl}
                    alt="Grad-CAM Heatmap"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "dual_saliency" && dualSaliencyUrl ? (
                  <img
                    src={dualSaliencyUrl}
                    alt="Dual Saliency Maps"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "difference" && differenceUrl ? (
                  <img
                    src={differenceUrl}
                    alt="Difference Heatmap"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "saliency_diff" && saliencyDiffUrl ? (
                  <img
                    src={saliencyDiffUrl}
                    alt="Saliency Difference"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "stroke_overlay" && strokeOverlayUrl ? (
                  <img
                    src={strokeOverlayUrl}
                    alt="Stroke Overlay"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "preprocessed" && preprocessedUrl ? (
                  <img
                    src={preprocessedUrl}
                    alt="Preprocessing Steps"
                    className="max-w-full max-h-96 object-contain relative z-10"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "normalized_overlay" && normalizedOverlayUrl ? (
                  <img
                    src={normalizedOverlayUrl}
                    alt="Normalized Overlay"
                    className="max-w-full max-h-96 object-contain relative z-10"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                ) : showHeatmap && heatmapType === "align_preview" && alignPreviewUrl ? (
                  <img
                    src={alignPreviewUrl}
                    alt="Alignment Preview"
                    className="max-w-full max-h-96 object-contain relative z-10"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                ) : signature2Url ? (
                  <img
                    src={signature2Url}
                    alt="Overlay"
                    className="max-w-full max-h-64 object-contain relative z-10"
                    style={{ opacity: overlay / 100, transform: `scale(${zoom / 100})` }}
                  />
                ) : null}
              </div>

              {signature2Url && (
                <div className="border rounded-lg p-4 bg-card flex items-center justify-center overflow-hidden">
                  <img
                    src={signature2Url}
                    alt="Signature 2"
                    className="max-w-full max-h-64 object-contain"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                </div>
              )}
            </div>
          )}

          {/* Controls */}
          {(signature1Url || signature2Url) && (
            <div className="grid md:grid-cols-2 gap-4 mb-8 p-4 bg-card border rounded-lg">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">Overlay Opacity</label>
                  <span className="text-sm text-muted-foreground">{overlay}%</span>
                </div>
                <div className="flex gap-2">
                  <Slider
                    value={[overlay]}
                    onValueChange={(value) => setOverlay(value[0])}
                    min={0}
                    max={100}
                    step={1}
                    className="flex-1"
                  />
                  {showHeatmap && heatmapType && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleRefreshVisualization}
                      disabled={
                        isGeneratingHeatmap ||
                        isGeneratingGradCam ||
                        isGeneratingDualSaliency ||
                        isGeneratingDifference ||
                        isGeneratingSaliencyDiff ||
                        isGeneratingStrokeOverlay ||
                        isGeneratingPreprocessed
                      }
                      title="Refresh visualization with new opacity"
                    >
                      Refresh
                    </Button>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">Zoom</label>
                  <span className="text-sm text-muted-foreground">{zoom}%</span>
                </div>
                <Slider
                  value={[zoom]}
                  onValueChange={(value) => setZoom(value[0])}
                  min={50}
                  max={200}
                  step={5}
                />
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 mb-8">
            <Button
              onClick={handleVerify}
              disabled={!signature1 || !signature2 || isVerifying}
              className="flex-1 md:flex-none"
            >
              <CheckCircle2 className="w-4 h-4 mr-2" />
              {isVerifying ? "Verifying..." : "Verify Signatures"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateHeatmap}
              disabled={!signature1 || !signature2 || isGeneratingHeatmap}
            >
              <Flame className="w-4 h-4 mr-2" />
              {isGeneratingHeatmap ? "Generating..." : "Saliency Heatmap"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateGradCam}
              disabled={!signature1 || !signature2 || isGeneratingGradCam}
            >
              <Flame className="w-4 h-4 mr-2" />
              {isGeneratingGradCam ? "Generating..." : "Grad-CAM"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateDualSaliency}
              disabled={!signature1 || !signature2 || isGeneratingDualSaliency}
            >
              <Flame className="w-4 h-4 mr-2" />
              {isGeneratingDualSaliency ? "Generating..." : "Dual Saliency"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateDifference}
              disabled={!signature1 || !signature2 || isGeneratingDifference}
            >
              <Flame className="w-4 h-4 mr-2" />
              {isGeneratingDifference ? "Generating..." : "Difference Map"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateSaliencyDiff}
              disabled={!signature1 || !signature2 || isGeneratingSaliencyDiff}
            >
              <Flame className="w-4 h-4 mr-2" />
              {isGeneratingSaliencyDiff ? "Generating..." : "Saliency Diff"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateStrokeOverlay}
              disabled={!signature1 || !signature2 || isGeneratingStrokeOverlay}
              className="bg-yellow-50 dark:bg-yellow-900/20 border-yellow-300 dark:border-yellow-700"
            >
              <CheckCircle2 className="w-4 h-4 mr-2" />
              {isGeneratingStrokeOverlay ? "Generating..." : "🖋️ Stroke Overlay"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGeneratePreprocessed}
              disabled={!signature1 || !signature2 || isGeneratingPreprocessed}
              className="bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700"
            >
              <CheckCircle2 className="w-4 h-4 mr-2" />
              {isGeneratingPreprocessed ? "Generating..." : "📊 Preprocessing"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateNormalizedOverlay}
              disabled={!signature1 || !signature2 || isGeneratingNormalizedOverlay}
              className="bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700"
            >
              <CheckCircle2 className="w-4 h-4 mr-2" />
              {isGeneratingNormalizedOverlay ? "Generating..." : "🎯 Normalized Overlay"}
            </Button>
            <Button
              variant="outline"
              onClick={handleGenerateAlignPreview}
              disabled={!signature1 || !signature2 || isGeneratingAlignPreview}
              className="bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700"
            >
              <CheckCircle2 className="w-4 h-4 mr-2" />
              {isGeneratingAlignPreview ? "Generating..." : "🔍 Alignment Preview"}
            </Button>
            <Button
              variant="outline"
              onClick={handleDownloadReport}
              disabled={!signature1 || !signature2 || isDownloadingReport}
            >
              <Download className="w-4 h-4 mr-2" />
              {isDownloadingReport ? "Downloading..." : "Download Report"}
            </Button>
            {(heatmapUrl || gradcamUrl || dualSaliencyUrl || differenceUrl || saliencyDiffUrl || strokeOverlayUrl || preprocessedUrl || normalizedOverlayUrl || alignPreviewUrl) && (
              <Button
                variant="outline"
                onClick={() => setShowHeatmap(!showHeatmap)}
              >
                {showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
              </Button>
            )}
          </div>

          {/* Loading */}
          {isVerifying && (
            <div className="space-y-4 p-6 rounded-lg bg-card border border-border mb-8">
              <p className="text-sm text-muted-foreground">Analyzing signatures...</p>
              <Progress value={66} />
            </div>
          )}

          {/* Results */}
          {result && !isVerifying && (
            <>
              {/* PREPROCESSING STATUS CARD - Shows alignment, resizing, RGB conversion */}
              {result.processing_info && (
                <div className="p-4 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 border-2 border-blue-300 dark:border-blue-700 mb-6">
                  <h3 className="text-sm font-semibold mb-3 text-blue-900 dark:text-blue-100 flex items-center gap-2">
                    🔧 5-Stage Preprocessing Pipeline Status
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
                    <div className="flex flex-col">
                      <span className="text-muted-foreground mb-1">Noise Removed:</span>
                      <span className={`font-semibold ${result.processing_info.noise_removed ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {result.processing_info.noise_removed ? '✅ Yes' : '❌ No'}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-muted-foreground mb-1">Baseline Aligned:</span>
                      <span className={`font-semibold ${result.processing_info.baseline_aligned ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {result.processing_info.baseline_aligned ? '✅ Yes' : '❌ No'}
                      </span>
                      {result.processing_info.baseline_aligned && (
                        <span className={`text-xs mt-1 ${result.processing_info.baseline_diff_px < 5 ? 'text-green-600' : 'text-yellow-600'}`}>
                          Diff: {result.processing_info.baseline_diff_px.toFixed(1)}px
                        </span>
                      )}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-muted-foreground mb-1">Size Normalized:</span>
                      <span className={`font-semibold ${result.processing_info.size_normalized ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {result.processing_info.size_normalized ? '✅ Yes (220x155)' : '❌ No'}
                      </span>
                      <span className="text-xs mt-1 text-muted-foreground">RGB format</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-muted-foreground mb-1">Brightness Matched:</span>
                      <span className={`font-semibold ${result.processing_info.brightness_matched ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {result.processing_info.brightness_matched ? '✅ Yes' : '❌ No'}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-muted-foreground mb-1">Alignment Quality:</span>
                      <span className={`font-semibold ${
                        result.processing_info.baseline_diff_px < 3 ? 'text-green-600' : 
                        result.processing_info.baseline_diff_px < 5 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {result.processing_info.baseline_diff_px < 3 ? '✅ Perfect' : 
                         result.processing_info.baseline_diff_px < 5 ? '⚠️ Good' : '❌ Poor'}
                      </span>
                    </div>
                  </div>
                  {result.processing_info.baseline_diff_px >= 5 && (
                    <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-3 p-2 bg-yellow-100 dark:bg-yellow-900 rounded">
                      ⚠️ Baseline alignment gap is {result.processing_info.baseline_diff_px.toFixed(1)}px - signatures may not align perfectly. This can cause false results.
                    </p>
                  )}
                  <p className="text-xs text-blue-700 dark:text-blue-300 mt-2">
                    💡 Click "📊 Preprocessing" button below to see visual pipeline (Original → Cleaned → Aligned → Resized → Overlay)
                  </p>
                </div>
              )}
              
              <div className="grid md:grid-cols-3 gap-6 mb-8">
              <div className="p-6 rounded-lg bg-card border border-border space-y-2">
                  <p className="text-sm text-muted-foreground">Combined Score</p>
                  <p className="text-4xl font-bold">
                    {result.detailed_metrics?.combined_score 
                      ? (result.detailed_metrics.combined_score * 100).toFixed(1)
                      : (result.similarity_score * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-muted-foreground">Multi-signal verification</p>
              </div>

              <div className="p-6 rounded-lg bg-card border border-border space-y-2">
                <p className="text-sm text-muted-foreground">Verdict</p>
                  <p className={`text-3xl font-bold ${getVerdictColor()}`}>{getVerdictText()}</p>
                  {result.similarity_score >= threshold ? (
                    <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
                )}
              </div>

              <div className="p-6 rounded-lg bg-card border border-border space-y-2">
                <p className="text-sm text-muted-foreground">Model Accuracy</p>
                  <p className="text-4xl font-bold">
                    {metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(0)}%` : "91%"}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Threshold: {(threshold * 100).toFixed(0)}%
                  </p>
                </div>
              </div>

              {/* Detailed Metrics - Handwriting & Stroke Analysis */}
              {result.detailed_metrics && (
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  {/* Multi-Signal Breakdown */}
                  <div className="p-6 rounded-lg bg-card border border-border">
                    <h3 className="text-lg font-semibold mb-4">Verification Signals</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Cosine Similarity ({((result.detailed_metrics.cosine_weight || 0.30) * 100).toFixed(0)}%):</span>
                        <span className="font-medium">{(result.detailed_metrics.cosine * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">SSIM ({((result.detailed_metrics.ssim_weight || 0.30) * 100).toFixed(0)}%):</span>
                        <span className="font-medium">{(result.detailed_metrics.ssim * 100).toFixed(1)}%</span>
                      </div>
                      {result.detailed_metrics.handwriting_score !== null && result.detailed_metrics.handwriting_score !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">🖋️ Handwriting Score ({((result.detailed_metrics.handwriting_weight || 0.10) * 100).toFixed(0)}%):</span>
                          <span className="font-medium">{(result.detailed_metrics.handwriting_score * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {result.detailed_metrics.stroke_comparison_score !== null && result.detailed_metrics.stroke_comparison_score !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">🖋️ Stroke Comparison ({((result.detailed_metrics.stroke_comparison_weight || 0.30) * 100).toFixed(0)}%):</span>
                          <span className="font-medium">{(result.detailed_metrics.stroke_comparison_score * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {result.detailed_metrics.stroke_similarity !== null && result.detailed_metrics.stroke_similarity !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-sm text-muted-foreground">🖋️ Stroke Similarity (component):</span>
                          <span className="font-medium">{(result.detailed_metrics.stroke_similarity * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      <div className="border-t pt-3 mt-3">
                        <div className="flex justify-between">
                          <span className="text-sm font-semibold">Combined Score:</span>
                          <span className="text-lg font-bold">{(result.detailed_metrics.combined_score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Handwriting Flow Analysis */}
                  {result.detailed_metrics.handwriting_flow1 && result.detailed_metrics.handwriting_flow2 && (
                    <div className="p-6 rounded-lg bg-card border border-border">
                      <h3 className="text-lg font-semibold mb-4">🖋️ Handwriting Analysis</h3>
                      <div className="space-y-3">
                        <div>
                          <span className="text-sm text-muted-foreground">Writing Style:</span>
                          <div className="flex gap-4 mt-1">
                            <span className="text-sm font-medium">
                              Sig1: {result.detailed_metrics.handwriting_flow1.writing_style || 'unknown'}
                            </span>
                            <span className="text-sm font-medium">
                              Sig2: {result.detailed_metrics.handwriting_flow2.writing_style || 'unknown'}
                            </span>
                          </div>
                        </div>
                        <div>
                          <span className="text-sm text-muted-foreground">Flow Smoothness:</span>
                          <div className="flex gap-4 mt-1">
                            <span className="text-sm font-medium">
                              Sig1: {(result.detailed_metrics.handwriting_flow1.flow_smoothness * 100).toFixed(0)}%
                            </span>
                            <span className="text-sm font-medium">
                              Sig2: {(result.detailed_metrics.handwriting_flow2.flow_smoothness * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                        <div>
                          <span className="text-sm text-muted-foreground">Stroke Count:</span>
                          <div className="flex gap-4 mt-1">
                            <span className="text-sm font-medium">
                              Sig1: {result.detailed_metrics.handwriting_flow1.stroke_count} strokes
                            </span>
                            <span className="text-sm font-medium">
                              Sig2: {result.detailed_metrics.handwriting_flow2.stroke_count} strokes
                            </span>
                          </div>
                        </div>
                        {result.detailed_metrics.stroke_comparison && (
                          <div className="border-t pt-3 mt-3 space-y-2">
                            <p className="text-sm font-semibold">Stroke Comparison:</p>
                            <div className="text-xs space-y-1">
                              <div className="flex justify-between">
                                <span>Count Similarity:</span>
                                <span>{(result.detailed_metrics.stroke_comparison.stroke_count_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Length Similarity:</span>
                                <span>{(result.detailed_metrics.stroke_comparison.stroke_length_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Direction Similarity:</span>
                                <span>{(result.detailed_metrics.stroke_comparison.stroke_direction_similarity * 100).toFixed(0)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Pressure Similarity:</span>
                                <span>{(result.detailed_metrics.stroke_comparison.stroke_pressure_similarity * 100).toFixed(0)}%</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Signature Detection Section */}
        <div className="border rounded-lg p-6 bg-card mb-8 mt-8 max-w-5xl mx-auto">
          <h3 className="text-lg font-semibold mb-4">🔍 Detect Signatures in Document</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Upload a document image to automatically detect and crop signature regions
          </p>
          <div className="flex gap-4">
            <div
              onDrop={(e) => {
                e.preventDefault();
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith("image/")) {
                  handleDetectSignatures(file);
                }
              }}
              onDragOver={(e) => e.preventDefault()}
              className="relative border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-primary transition-colors cursor-pointer bg-card flex-1"
            >
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    handleDetectSignatures(file);
                  }
                }}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <FileImage className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-sm font-medium mb-1">Drop document image here</p>
              <p className="text-xs text-muted-foreground">Auto-detect signature locations</p>
            </div>
            {isDetectingSignatures && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
                Detecting...
              </div>
            )}
          </div>
          {detectedSignatures && detectedSignatures.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-medium mb-2">
                Found {detectedSignatures.length} signature(s):
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {detectedSignatures.map((sig: any, idx: number) => (
                  <div key={idx} className="border rounded p-2 bg-card">
                    <img
                      src={sig.thumbnail || sig.image_crop}
                      alt={`Signature ${idx + 1}`}
                      className="w-full h-auto rounded mb-2"
                    />
                    <p className="text-xs text-muted-foreground">
                      Confidence: {(sig.confidence * 100).toFixed(0)}%
                    </p>
                    <Button
                      size="sm"
                      variant="outline"
                      className="w-full mt-2 text-xs"
                      onClick={() => {
                        toast({
                          title: "Signature selected",
                          description: `Signature ${idx + 1} selected. Use file selector to set as Signature 1 or 2.`,
                        });
                      }}
                    >
                      Use as Sig {idx === 0 ? "1" : "2"}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Security Disclaimer */}
        {disclaimer && (
          <div className="mt-8 p-4 rounded-lg bg-muted/50 border border-border max-w-5xl mx-auto">
            <p className="text-sm text-muted-foreground">
              <strong>🔒 Security Notice:</strong> {disclaimer}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Verify;
