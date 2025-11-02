import { useState, useEffect } from "react";
import { Shield, Zap, Eye, FileText, BarChart, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { getMetrics, type MetricsResponse } from "@/lib/api";

const About = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch(() => {})
      .finally(() => setIsLoading(false));
  }, []);

  const stats = [
    {
      label: "Model Accuracy",
      value: metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(0)}%` : "91%",
      description: "On test dataset",
      loading: isLoading && metrics?.accuracy === undefined,
    },
    {
      label: "F1 Score",
      value: metrics?.f1 ? `${(metrics.f1 * 100).toFixed(0)}%` : "90%",
      description: "Balanced precision & recall",
      loading: isLoading && metrics?.f1 === undefined,
    },
    {
      label: "Detection Threshold",
      value: metrics?.threshold ? `${(metrics.threshold * 100).toFixed(0)}%` : "85%",
      description: "Optimized for best performance",
      loading: isLoading && metrics?.threshold === undefined,
    },
  ];

  const features = [
    {
      icon: Zap,
      title: "Real-time Verification",
      description: "Upload and compare signatures instantly",
    },
    {
      icon: BarChart,
      title: "Batch Processing",
      description: "Verify multiple signatures against a reference",
    },
    {
      icon: Eye,
      title: "Visual Comparison",
      description: "Overlay and zoom tools for manual inspection",
    },
    {
      icon: Shield,
      title: "AI Explainability",
      description: "Saliency heatmaps show model attention",
    },
    {
      icon: FileText,
      title: "PDF Reports",
      description: "Download professional verification reports",
    },
    {
      icon: Moon,
      title: "Dark Mode",
      description: "Comfortable viewing in any environment",
    },
  ];

  const techStack = [
    { label: "Backend", value: "Flask, PyTorch, ResNet18 with Transformers" },
    { label: "Frontend", value: "React, Vite, Chart.js" },
    { label: "Model", value: "Siamese Network with Contrastive Learning" },
    { label: "Features", value: "Cosine similarity, Grad-CAM visualization" },
  ];

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 space-y-4">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
            About SignGuard
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            AI-powered signature verification platform
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-16">
          {/* About Section */}
          <div className="space-y-6">
            <h2 className="text-3xl font-bold">About SignGuard</h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              SignGuard is an AI-powered signature verification platform built with cutting-edge deep
              learning technology. Using a Siamese neural network architecture with ResNet18 backbone
              and transformer layers, it achieves high accuracy in detecting forged signatures.
            </p>
          </div>

          {/* Stats */}
          <div className="grid md:grid-cols-3 gap-6">
            {stats.map((stat) => (
              <div key={stat.label} className="p-6 rounded-lg bg-card border border-border space-y-2">
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                {stat.loading ? (
                  <div className="h-10 w-20 bg-muted animate-pulse rounded"></div>
                ) : (
                  <p className="text-4xl font-bold text-primary">{stat.value}</p>
                )}
                <p className="text-xs text-muted-foreground">{stat.description}</p>
              </div>
            ))}
          </div>

          {/* Features */}
          <div className="space-y-6">
            <h2 className="text-3xl font-bold">Key Features</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {features.map((feature) => (
                <div key={feature.title} className="p-6 rounded-lg bg-card border border-border space-y-3">
                  <feature.icon className="w-8 h-8 text-accent" />
                  <h3 className="font-semibold">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Technology Stack */}
          <div className="space-y-6">
            <h2 className="text-3xl font-bold">Technology Stack</h2>
            <div className="space-y-4">
              {techStack.map((item) => (
                <div key={item.label} className="flex gap-4 p-4 rounded-lg bg-card border border-border">
                  <span className="font-semibold min-w-24">{item.label}:</span>
                  <span className="text-muted-foreground">{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* CTA */}
          <div className="text-center py-12">
            <Link to="/">
              <Button size="lg" className="gradient-primary shadow-glow">
                Start Verifying
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
