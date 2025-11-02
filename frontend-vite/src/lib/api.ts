// API Configuration
// Try to use environment variable, fallback to default
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

// Export for debugging
export const API_BASE_URL = API_BASE;

// Helper to handle fetch errors with better messages
async function handleFetch<T>(url: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      // Try to get error message from response
      let errorMessage = `Server error: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch {
        // If not JSON, use status text
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }
    
    return response.json();
  } catch (error: any) {
    // Handle network errors (CORS, connection refused, etc.)
    if (error.name === 'TypeError' && (error.message.includes('fetch') || error.message.includes('Failed to fetch'))) {
      throw new Error(
        `🔴 Connection Failed: Cannot reach backend at ${API_BASE}\n` +
        `Possible causes:\n` +
        `1. Flask server not running (start with: python -m backend.app)\n` +
        `2. Wrong port (expected: 5000)\n` +
        `3. CORS blocked (check browser console)\n` +
        `4. Firewall blocking localhost\n` +
        `Original error: ${error.message}`
      );
    }
    throw error;
  }
}

async function handleFetchBlob(url: string, options?: RequestInit): Promise<Blob> {
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      let errorMessage = `Server error: ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch {
        errorMessage = response.statusText || errorMessage;
      }
      throw new Error(errorMessage);
    }
    
    return response.blob();
  } catch (error: any) {
    if (error.name === 'TypeError' && (error.message.includes('fetch') || error.message.includes('Failed to fetch'))) {
      throw new Error(
        `🔴 Connection Failed: Cannot reach backend at ${API_BASE}\n` +
        `Make sure Flask server is running: python -m backend.app\n` +
        `Original error: ${error.message}`
      );
    }
    throw error;
  }
}

export interface PredictResponse {
  similarity_score: number;
  verdict: string;
  processing_info?: {
    baseline_aligned: boolean;
    baseline_diff_px: number;
    noise_removed: boolean;
    size_normalized: boolean;
    brightness_matched: boolean;
  };
  detailed_metrics?: {
    cosine: number;
    cosine_weight?: number;
    orb_ratio: number;
    orb_weight?: number;
    orb_matches: number;
    ssim: number;
    ssim_weight?: number;
    combined_score: number;
    confidence: string;
    requires_review: boolean;
    safety_flags: string[];
    stroke_similarity?: number;
    handwriting_score?: number;
    handwriting_weight?: number;
    handwriting_details?: {
      stroke_similarity?: number;
      flow_similarity?: number;
      style_match?: number;
      count_similarity?: number;
      length_similarity?: number;
      direction_similarity?: number;
      pressure_similarity?: number;
    };
    handwriting_flow1?: {
      writing_style: string;
      flow_smoothness: number;
      stroke_count: number;
    };
    handwriting_flow2?: {
      writing_style: string;
      flow_smoothness: number;
      stroke_count: number;
    };
    stroke_comparison?: {
      stroke_count_similarity: number;
      stroke_length_similarity: number;
      stroke_direction_similarity: number;
      stroke_pressure_similarity: number;
      overall_stroke_similarity: number;
    };
    stroke_comparison_score?: number;
    stroke_comparison_weight?: number;
  };
}

export interface BatchResult {
  filename: string;
  similarity_score?: number;
  verdict?: string;
  error?: string;
}

export interface BatchResponse {
  results: BatchResult[];
}

export interface HistoryItem {
  type: 'single' | 'batch';
  timestamp: string;
  result?: PredictResponse;
  count?: number;
  results?: BatchResult[];
}

export interface HistoryResponse {
  history: HistoryItem[];
}

export interface MetricsResponse {
  accuracy?: number | null;
  f1?: number | null;
  threshold?: number;
}

// Verify signatures
export async function verifySignatures(img1: File, img2: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);

  return handleFetch<PredictResponse>(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });
}

// Batch verification
export async function batchVerify(reference: File, files: FileList): Promise<BatchResponse> {
  const formData = new FormData();
  formData.append('reference', reference);
  
  Array.from(files).forEach(file => {
    formData.append('files', file);
  });

  return handleFetch<BatchResponse>(`${API_BASE}/batch_predict`, {
    method: 'POST',
    body: formData,
  });
}

// Generate saliency heatmap
export async function generateSaliencyHeatmap(img1: File, img2: File, opacity: number = 0.5): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('opacity', opacity.toString());

  return handleFetchBlob(`${API_BASE}/saliency`, {
    method: 'POST',
    body: formData,
  });
}

// Generate dual saliency maps (both signatures side by side)
export async function generateDualSaliency(img1: File, img2: File, opacity: number = 0.5): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('opacity', opacity.toString());

  return handleFetchBlob(`${API_BASE}/dual_saliency`, {
    method: 'POST',
    body: formData,
  });
}

// Generate difference heatmap (pixel differences)
export async function generateDifferenceHeatmap(img1: File, img2: File, opacity: number = 0.6): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('opacity', opacity.toString());

  return handleFetchBlob(`${API_BASE}/difference`, {
    method: 'POST',
    body: formData,
  });
}

// Generate saliency difference heatmap
export async function generateSaliencyDifference(img1: File, img2: File, opacity: number = 0.6): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('opacity', opacity.toString());

  return handleFetchBlob(`${API_BASE}/saliency_diff`, {
    method: 'POST',
    body: formData,
  });
}

// Generate Grad-CAM heatmap (supports single or dual images)
export async function generateGradCamHeatmap(img1: File, img2?: File, opacity?: number): Promise<Blob> {
  const formData = new FormData();
  
  const opacityValue = opacity ?? 0.5; // Default to 0.5 if not provided
  
  if (img2) {
    // Dual image mode (for signature comparison)
    formData.append('img1', img1);
    formData.append('img2', img2);
  } else {
    // Single image mode (legacy)
    formData.append('img', img1);
  }
  formData.append('opacity', opacityValue.toString());

  return handleFetchBlob(`${API_BASE}/gradcam`, {
    method: 'POST',
    body: formData,
  });
}

// Get disclaimer
export async function getDisclaimer(): Promise<{
  disclaimer: string;
  security_practices: string[];
  data_handling: {
    images_stored: boolean;
    embeddings_stored: boolean;
    metadata_only: boolean;
  };
}> {
  return handleFetch(`${API_BASE}/disclaimer`);
}

// Download report
export async function downloadReport(
  img1: File,
  img2: File,
  heatmap?: Blob
): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  
  if (heatmap) {
    formData.append('heatmap', heatmap, 'heatmap.png');
  }

  return handleFetchBlob(`${API_BASE}/report`, {
    method: 'POST',
    body: formData,
  });
}

// Get history
export async function getHistory(): Promise<HistoryResponse> {
  return handleFetch<HistoryResponse>(`${API_BASE}/history`);
}

// Get metrics
export async function getMetrics(): Promise<MetricsResponse> {
  return handleFetch<MetricsResponse>(`${API_BASE}/metrics`);
}

// Health check
export async function healthCheck(): Promise<{ status: string }> {
  return handleFetch<{ status: string }>(`${API_BASE}/health`);
}

// Get visualization explanation
export interface VizExplanation {
  title: string;
  description: string;
  interpretation: string;
  color_legend: string;
}

export async function getVizExplanation(vizType: string): Promise<VizExplanation> {
  return handleFetch<VizExplanation>(`${API_BASE}/viz_explanation/${vizType}`);
}

// Generate stroke overlay (shows which strokes match)
export async function generateStrokeOverlay(img1: File, img2: File, opacity: number = 0.5): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('opacity', opacity.toString());

  return handleFetchBlob(`${API_BASE}/stroke_overlay`, {
    method: 'POST',
    body: formData,
  });
}

// Get preprocessed preview (shows noise removal and alignment steps)
export async function getPreprocessedPreview(img1: File, img2: File): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);

  return handleFetchBlob(`${API_BASE}/preprocessed_preview`, {
    method: 'POST',
    body: formData,
  });
}

// 5-Stage Normalization Pipeline APIs
export interface DetectedSignature {
  id: number;
  bbox: [number, number, number, number];
  confidence: number;
  thumbnail: string;
  area: number;
}

export async function detectSignaturesInDocument(image: File): Promise<{
  signatures_found: number;
  signatures: DetectedSignature[];
  message: string;
}> {
  const formData = new FormData();
  formData.append('image', image);
  return handleFetch(`${API_BASE}/detect_signatures_multi`, {
    method: 'POST',
    body: formData,
  });
}

export async function getNormalizedOverlay(
  img1: File,
  img2: File,
  options?: {
    show_baseline?: boolean;
    enable_baseline_align?: boolean;
    enable_brightness_match?: boolean;
    opacity?: number;
  }
): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('show_baseline', (options?.show_baseline ?? true).toString());
  formData.append('enable_baseline_align', (options?.enable_baseline_align ?? true).toString());
  formData.append('enable_brightness_match', (options?.enable_brightness_match ?? true).toString());
  formData.append('opacity', (options?.opacity ?? 0.5).toString());
  return handleFetchBlob(`${API_BASE}/normalized_overlay`, { method: 'POST', body: formData });
}

// Generate align preview (shows alignment visualization)
export async function generateAlignPreview(
  img1: File,
  img2: File,
  options?: {
    use_advanced?: boolean;
    opacity?: number;
  }
): Promise<Blob> {
  const formData = new FormData();
  formData.append('img1', img1);
  formData.append('img2', img2);
  formData.append('use_advanced', (options?.use_advanced ?? true).toString());
  formData.append('opacity', (options?.opacity ?? 0.5).toString());
  return handleFetchBlob(`${API_BASE}/align_preview`, { method: 'POST', body: formData });
}

// Detect signatures in document (single image detection)
export async function detectSignatures(image: File): Promise<{
  signatures_found: number;
  signatures: Array<{
    index: number;
    bounding_box: { x: number; y: number; width: number; height: number };
    thumbnail: string;
    area: number;
  }>;
  message: string;
}> {
  const formData = new FormData();
  formData.append('img', image);
  return handleFetch(`${API_BASE}/detect_signatures`, {
    method: 'POST',
    body: formData,
  });
}

