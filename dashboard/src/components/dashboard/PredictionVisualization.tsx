/**
 * Prediction Visualization Component
 * Provides charts and analytics for prediction data
 */

'use client';

import React, { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell
} from 'recharts';

import { 
  TrendingUp, 
  BarChart3, 
  PieChart as PieChartIcon,
  Activity,
  Clock,
  Target
} from 'lucide-react';

import { formatNumber, formatDuration } from '@/utils/format';
import type { ModelInfo } from '@/types';

interface PredictionVisualizationProps {
  predictions: any[];
  modelInfo?: ModelInfo | null;
}

export function PredictionVisualization({ predictions, modelInfo }: PredictionVisualizationProps) {
  // Process data for visualizations
  const chartData = useMemo(() => {
    if (predictions.length === 0) return null;

    // Time series data for line chart
    const timeSeriesData = predictions
      .slice()
      .reverse()
      .map((pred, index) => ({
        index: index + 1,
        prediction: pred.prediction / 100000, // Convert to hundreds of thousands
        processingTime: pred.processing_time_ms,
        timestamp: new Date(pred.timestamp).getTime()
      }));

    // Distribution data for histogram
    const predictionValues = predictions.map(p => p.prediction / 100000);
    const min = Math.min(...predictionValues);
    const max = Math.max(...predictionValues);
    const binCount = Math.min(10, Math.ceil(Math.sqrt(predictions.length)));
    const binSize = (max - min) / binCount;
    
    const distributionData = Array.from({ length: binCount }, (_, i) => {
      const binStart = min + i * binSize;
      const binEnd = binStart + binSize;
      const count = predictionValues.filter(v => v >= binStart && v < binEnd).length;
      return {
        range: `$${binStart.toFixed(0)}K-${binEnd.toFixed(0)}K`,
        count,
        binStart,
        binEnd
      };
    });

    // Processing time distribution
    const processingTimes = predictions.map(p => p.processing_time_ms);
    const avgProcessingTime = processingTimes.reduce((sum, time) => sum + time, 0) / processingTimes.length;
    const maxProcessingTime = Math.max(...processingTimes);
    const minProcessingTime = Math.min(...processingTimes);

    const performanceData = [
      { name: 'Average', value: avgProcessingTime, color: '#3b82f6' },
      { name: 'Fastest', value: minProcessingTime, color: '#10b981' },
      { name: 'Slowest', value: maxProcessingTime, color: '#ef4444' }
    ];

    // Scatter plot data (prediction vs processing time)
    const scatterData = predictions.map((pred, index) => ({
      x: pred.prediction / 100000,
      y: pred.processing_time_ms,
      index: index + 1
    }));

    return {
      timeSeries: timeSeriesData,
      distribution: distributionData,
      performance: performanceData,
      scatter: scatterData,
      stats: {
        total: predictions.length,
        average: predictionValues.reduce((sum, val) => sum + val, 0) / predictionValues.length,
        min: min,
        max: max,
        avgProcessingTime,
        minProcessingTime,
        maxProcessingTime
      }
    };
  }, [predictions]);

  if (!chartData) {
    return (
      <Card>
        <CardContent className="p-8 text-center text-muted-foreground">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No data to visualize</h3>
          <p className="text-sm">
            Make some predictions to see analytics and charts
          </p>
        </CardContent>
      </Card>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded-lg shadow-lg">
          <p className="font-medium">{`Prediction #${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {
                entry.name === 'Processing Time' 
                  ? formatDuration(entry.value)
                  : `$${formatNumber(entry.value, 1)}K`
              }
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-600" />
              <div>
                <div className="text-sm font-medium">Average Prediction</div>
                <div className="text-lg font-bold">
                  ${formatNumber(chartData.stats.average, 1)}K
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-green-600" />
              <div>
                <div className="text-sm font-medium">Range</div>
                <div className="text-lg font-bold">
                  ${formatNumber(chartData.stats.min, 1)}K - ${formatNumber(chartData.stats.max, 1)}K
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-orange-600" />
              <div>
                <div className="text-sm font-medium">Avg Processing</div>
                <div className="text-lg font-bold">
                  {formatDuration(chartData.stats.avgProcessingTime)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-purple-600" />
              <div>
                <div className="text-sm font-medium">Total Predictions</div>
                <div className="text-lg font-bold">
                  {chartData.stats.total}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Performance Info */}
      {modelInfo && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Model Performance</CardTitle>
            <CardDescription>
              Current model metrics and information
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <div className="text-sm font-medium text-muted-foreground">RMSE</div>
                <div className="text-2xl font-bold">
                  {formatNumber(modelInfo.performance_metrics.rmse, 4)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-muted-foreground">MAE</div>
                <div className="text-2xl font-bold">
                  {formatNumber(modelInfo.performance_metrics.mae, 4)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-muted-foreground">R² Score</div>
                <div className="text-2xl font-bold">
                  {formatNumber(modelInfo.performance_metrics.r2, 3)}
                </div>
              </div>
            </div>
            <div className="mt-4 flex items-center space-x-2">
              <Badge variant={modelInfo.gpu_accelerated ? 'default' : 'secondary'}>
                {modelInfo.model_type}
              </Badge>
              <Badge variant="outline">
                v{modelInfo.model_version}
              </Badge>
              {modelInfo.gpu_accelerated && (
                <Badge variant="default" className="bg-green-600">
                  GPU Accelerated
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      <Tabs defaultValue="timeline" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="scatter">Correlation</TabsTrigger>
        </TabsList>

        <TabsContent value="timeline">
          <Card>
            <CardHeader>
              <CardTitle>Prediction Timeline</CardTitle>
              <CardDescription>
                Prediction values over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.timeSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="index" 
                      label={{ value: 'Prediction #', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'Value ($K)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Line 
                      type="monotone" 
                      dataKey="prediction" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                      name="Prediction Value"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="distribution">
          <Card>
            <CardHeader>
              <CardTitle>Value Distribution</CardTitle>
              <CardDescription>
                Distribution of predicted house values
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData.distribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="range" 
                      angle={-45}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis 
                      label={{ value: 'Count', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle>Processing Performance</CardTitle>
              <CardDescription>
                Processing time statistics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData.performance}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${formatDuration(value || 0)}`}
                    >
                      {chartData.performance.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => formatDuration(value)} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scatter">
          <Card>
            <CardHeader>
              <CardTitle>Value vs Processing Time</CardTitle>
              <CardDescription>
                Correlation between prediction value and processing time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart data={chartData.scatter}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      type="number"
                      dataKey="x"
                      name="Prediction Value"
                      label={{ value: 'Value ($K)', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      type="number"
                      dataKey="y"
                      name="Processing Time"
                      label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      formatter={(value, name) => [
                        name === 'Processing Time' ? formatDuration(value as number) : `$${formatNumber(value as number, 1)}K`,
                        name
                      ]}
                    />
                    <Scatter dataKey="y" fill="#3b82f6" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}