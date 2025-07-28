/**
 * Performance Metrics Component
 * Displays model performance metrics and information
 */

'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Target, 
  TrendingUp, 
  Zap, 
  Calendar,
  Award,
  Info
} from 'lucide-react';

import { formatNumber, formatDate } from '@/utils/format';
import type { ModelInfo } from '@/types';

interface PerformanceMetricsProps {
  modelInfo?: ModelInfo | null;
}

export function PerformanceMetrics({ modelInfo }: PerformanceMetricsProps) {
  if (!modelInfo) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">Model information not available</p>
        </CardContent>
      </Card>
    );
  }

  // Calculate performance score (0-100) based on R² score
  const performanceScore = Math.max(0, Math.min(100, modelInfo.performance_metrics.r2 * 100));
  
  // Determine performance level
  const getPerformanceLevel = (r2: number) => {
    if (r2 >= 0.9) return { level: 'Excellent', color: 'bg-green-600' };
    if (r2 >= 0.8) return { level: 'Good', color: 'bg-blue-600' };
    if (r2 >= 0.7) return { level: 'Fair', color: 'bg-yellow-600' };
    return { level: 'Poor', color: 'bg-red-600' };
  };

  const performance = getPerformanceLevel(modelInfo.performance_metrics.r2);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Award className="h-5 w-5 mr-2" />
          Model Performance
        </CardTitle>
        <CardDescription>
          Current model metrics and capabilities
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Model Info */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-medium">{modelInfo.model_type}</h3>
            <p className="text-sm text-muted-foreground">
              Version {modelInfo.model_version}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant={modelInfo.gpu_accelerated ? 'default' : 'secondary'}>
              {modelInfo.gpu_accelerated ? 'GPU Accelerated' : 'CPU Only'}
            </Badge>
            {modelInfo.gpu_accelerated && (
              <Zap className="h-4 w-4 text-yellow-500" />
            )}
          </div>
        </div>

        {/* Performance Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Overall Performance</span>
            <Badge className={performance.color}>
              {performance.level}
            </Badge>
          </div>
          <Progress value={performanceScore} className="h-2" />
          <p className="text-xs text-muted-foreground">
            Based on R² score: {formatNumber(modelInfo.performance_metrics.r2, 3)}
          </p>
        </div>

        {/* Detailed Metrics */}
        <div className="grid gap-4">
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-600" />
              <div>
                <div className="text-sm font-medium">RMSE</div>
                <div className="text-xs text-muted-foreground">Root Mean Square Error</div>
              </div>
            </div>
            <div className="text-right">
              <div className="font-medium">
                {formatNumber(modelInfo.performance_metrics.rmse, 4)}
              </div>
              <div className="text-xs text-muted-foreground">
                Lower is better
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-green-600" />
              <div>
                <div className="text-sm font-medium">MAE</div>
                <div className="text-xs text-muted-foreground">Mean Absolute Error</div>
              </div>
            </div>
            <div className="text-right">
              <div className="font-medium">
                {formatNumber(modelInfo.performance_metrics.mae, 4)}
              </div>
              <div className="text-xs text-muted-foreground">
                Lower is better
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-purple-600" />
              <div>
                <div className="text-sm font-medium">R² Score</div>
                <div className="text-xs text-muted-foreground">Coefficient of Determination</div>
              </div>
            </div>
            <div className="text-right">
              <div className="font-medium">
                {formatNumber(modelInfo.performance_metrics.r2, 3)}
              </div>
              <div className="text-xs text-muted-foreground">
                Higher is better (max: 1.0)
              </div>
            </div>
          </div>
        </div>

        {/* Training Date */}
        <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
          <div className="flex items-center space-x-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Training Date</span>
          </div>
          <span className="text-sm">
            {formatDate(modelInfo.training_date)}
          </span>
        </div>

        {/* Performance Interpretation */}
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-medium text-blue-900 mb-1">
            Performance Interpretation
          </h4>
          <div className="text-xs text-blue-800 space-y-1">
            <p>
              <strong>R² Score ({formatNumber(modelInfo.performance_metrics.r2, 3)}):</strong> 
              {' '}Explains {formatNumber(modelInfo.performance_metrics.r2 * 100, 1)}% of the variance in house prices
            </p>
            <p>
              <strong>RMSE ({formatNumber(modelInfo.performance_metrics.rmse, 2)}):</strong> 
              {' '}Average prediction error of ${formatNumber(modelInfo.performance_metrics.rmse * 100000, 0)}
            </p>
            <p>
              <strong>MAE ({formatNumber(modelInfo.performance_metrics.mae, 2)}):</strong> 
              {' '}Typical prediction error of ${formatNumber(modelInfo.performance_metrics.mae * 100000, 0)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}