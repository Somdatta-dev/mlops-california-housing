/**
 * Real-Time Prediction Dashboard Component
 * Provides real-time prediction feed, interactive forms, and visualization
 */

'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Activity, 
  TrendingUp, 
  Clock, 
  Zap,
  AlertCircle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';

import { PredictionForm } from './PredictionForm';
import { RealTimePredictionFeed } from './RealTimePredictionFeed';
import { PredictionVisualization } from './PredictionVisualization';
import { PredictionHistory } from './PredictionHistory';
import { PerformanceMetrics } from './PerformanceMetrics';

import { useWebSocket, useRealtimePredictions } from '@/hooks/useWebSocket';
import { useModelInfo, useHealthStatus } from '@/hooks/useApi';
import { formatNumber, formatDuration, formatRelativeTime } from '@/utils/format';
import type { PredictionRequest } from '@/types';

interface PredictionData {
  prediction: number;
  processing_time_ms: number;
  model_version: string;
  timestamp: string;
  request_id?: string;
}

export function PredictionDashboard() {
  const { isConnected, error: wsError, reconnect } = useWebSocket();
  const { predictions, latestPrediction, clearPredictions } = useRealtimePredictions();
  const { data: modelInfo } = useModelInfo();
  const { data: healthStatus, loading: healthLoading } = useHealthStatus();

  const [activeTab, setActiveTab] = useState('predict');
  const [predictionStats, setPredictionStats] = useState({
    totalPredictions: 0,
    avgProcessingTime: 0,
    avgPredictionValue: 0,
    lastPredictionTime: null as string | null
  });

  // Update prediction statistics
  useEffect(() => {
    if (predictions.length > 0) {
      const total = predictions.length;
      const validPredictions = predictions.filter((p): p is PredictionData => p != null);
      const avgTime = validPredictions.reduce((sum, p) => sum + (p?.processing_time_ms || 0), 0) / total;
      const avgValue = validPredictions.reduce((sum, p) => sum + (p?.prediction || 0), 0) / total;
      const lastTime = validPredictions[0]?.timestamp;

      setPredictionStats({
        totalPredictions: total,
        avgProcessingTime: avgTime,
        avgPredictionValue: avgValue,
        lastPredictionTime: lastTime
      });
    }
  }, [predictions]);

  const handlePredictionSubmit = (request: PredictionRequest) => {
    // The prediction will be handled by the form component
    // and the result will come through WebSocket
    console.log('Prediction submitted:', request);
  };

  const ConnectionStatus = () => (
    <div className="flex items-center space-x-2">
      <div className={`w-2 h-2 rounded-full ${
        isConnected ? 'bg-green-500' : 'bg-red-500'
      }`} />
      <span className="text-sm text-muted-foreground">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
      {!isConnected && (
        <Button
          variant="outline"
          size="sm"
          onClick={reconnect}
          className="ml-2"
        >
          <RefreshCw className="h-3 w-3 mr-1" />
          Reconnect
        </Button>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Real-Time Predictions</h2>
          <p className="text-muted-foreground">
            Interactive prediction dashboard with live updates and analytics
          </p>
        </div>
        <ConnectionStatus />
      </div>

      {/* WebSocket Error Alert */}
      {wsError ? (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            WebSocket connection error: {typeof wsError === 'string' ? wsError : 'Connection failed'}
          </AlertDescription>
        </Alert>
      ) : null}

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{predictionStats.totalPredictions}</div>
            <p className="text-xs text-muted-foreground">
              Since connection started
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatDuration(predictionStats.avgProcessingTime)}
            </div>
            <p className="text-xs text-muted-foreground">
              Per prediction
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Prediction</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${formatNumber(predictionStats.avgPredictionValue / 100000, 1)}K
            </div>
            <p className="text-xs text-muted-foreground">
              House value estimate
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-2">
              {healthLoading ? (
                <div className="text-sm">Loading...</div>
              ) : healthStatus?.model_loaded ? (
                <>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium">Active</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  <span className="text-sm font-medium">Inactive</span>
                </>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {modelInfo?.model_type || 'Unknown model'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Latest Prediction Alert */}
      {latestPrediction ? (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            Latest prediction: ${formatNumber(((latestPrediction as PredictionData)?.prediction || 0) / 100000, 1)}K 
            ({formatDuration((latestPrediction as PredictionData)?.processing_time_ms || 0)} processing time)
            {predictionStats.lastPredictionTime && (
              <span className="ml-2 text-muted-foreground">
                • {formatRelativeTime(predictionStats.lastPredictionTime)}
              </span>
            )}
          </AlertDescription>
        </Alert>
      ) : null}

      {/* Main Dashboard Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="predict">Make Prediction</TabsTrigger>
          <TabsTrigger value="feed">Live Feed</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="predict" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <PredictionForm 
              onSubmit={handlePredictionSubmit}
              isConnected={isConnected}
              modelInfo={modelInfo}
            />
            <div className="space-y-4">
              <PerformanceMetrics modelInfo={modelInfo} />
              {latestPrediction ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Latest Result</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Prediction:</span>
                        <span className="font-medium">
                          ${formatNumber(((latestPrediction as PredictionData)?.prediction || 0) / 100000, 1)}K
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Processing Time:</span>
                        <span className="font-medium">
                          {formatDuration((latestPrediction as PredictionData)?.processing_time_ms || 0)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Model Version:</span>
                        <span className="font-medium">
                          {(latestPrediction as PredictionData)?.model_version || 'Unknown'}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : null}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="feed" className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">Real-Time Prediction Feed</h3>
            <Button
              variant="outline"
              size="sm"
              onClick={clearPredictions}
              disabled={predictions.length === 0}
            >
              Clear Feed
            </Button>
          </div>
          <RealTimePredictionFeed 
            predictions={predictions}
            isConnected={isConnected}
          />
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <PredictionVisualization 
            predictions={predictions}
            modelInfo={modelInfo}
          />
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          <PredictionHistory />
        </TabsContent>
      </Tabs>
    </div>
  );
}