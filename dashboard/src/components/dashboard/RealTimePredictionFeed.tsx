/**
 * Real-Time Prediction Feed Component
 * Displays live prediction updates with scrollable feed
 */

'use client';

import React, { useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Activity, 
  Clock, 
  DollarSign, 
  Zap,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';

import { formatNumber, formatDuration, formatRelativeTime } from '@/utils/format';
import type { PredictionResponse } from '@/types';

interface RealTimePredictionFeedProps {
  predictions: any[]; // Using any for now since we don't have exact WebSocket type
  isConnected: boolean;
}

interface PredictionItemProps {
  prediction: any;
  index: number;
  previousPrediction?: any;
}

function PredictionItem({ prediction, index, previousPrediction }: PredictionItemProps) {
  const isRecent = index < 3; // Highlight recent predictions
  const trend = previousPrediction 
    ? prediction.prediction > previousPrediction.prediction ? 'up' 
    : prediction.prediction < previousPrediction.prediction ? 'down' 
    : 'stable'
    : 'stable';

  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
  const trendColor = trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-gray-600';

  return (
    <div className={`p-4 border rounded-lg transition-all duration-300 ${
      isRecent ? 'bg-blue-50 border-blue-200' : 'bg-white border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Badge variant={isRecent ? 'default' : 'secondary'} className="text-xs">
            #{prediction.request_id?.slice(-8) || `${index + 1}`}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {formatRelativeTime(prediction.timestamp)}
          </span>
        </div>
        <div className="flex items-center space-x-1">
          <TrendIcon className={`h-3 w-3 ${trendColor}`} />
          <span className="text-sm font-medium">
            ${formatNumber(prediction.prediction / 100000, 1)}K
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
        <div className="flex items-center space-x-1">
          <Clock className="h-3 w-3" />
          <span>{formatDuration(prediction.processing_time_ms)}</span>
        </div>
        <div className="flex items-center space-x-1">
          <Zap className="h-3 w-3" />
          <span>{prediction.model_version}</span>
        </div>
      </div>

      {prediction.confidence_interval && (
        <div className="mt-2 text-xs text-muted-foreground">
          Confidence: ${formatNumber(prediction.confidence_interval[0] / 100000, 1)}K - ${formatNumber(prediction.confidence_interval[1] / 100000, 1)}K
        </div>
      )}
    </div>
  );
}

export function RealTimePredictionFeed({ predictions, isConnected }: RealTimePredictionFeedProps) {
  const feedRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = React.useState(true);

  // Auto-scroll to top when new predictions arrive
  useEffect(() => {
    if (autoScroll && feedRef.current && predictions.length > 0) {
      feedRef.current.scrollTop = 0;
    }
  }, [predictions, autoScroll]);

  // Handle manual scrolling
  const handleScroll = () => {
    if (feedRef.current) {
      const { scrollTop } = feedRef.current;
      setAutoScroll(scrollTop === 0);
    }
  };

  const getConnectionStatus = () => {
    if (isConnected) {
      return {
        icon: Activity,
        text: 'Connected - Live updates active',
        variant: 'default' as const,
        color: 'text-green-600'
      };
    } else {
      return {
        icon: AlertCircle,
        text: 'Disconnected - No live updates',
        variant: 'destructive' as const,
        color: 'text-red-600'
      };
    }
  };

  const status = getConnectionStatus();

  return (
    <div className="space-y-4">
      {/* Feed Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Live Prediction Feed</CardTitle>
              <CardDescription>
                Real-time updates of housing price predictions
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <status.icon className={`h-4 w-4 ${status.color}`} />
              <span className={`text-sm ${status.color}`}>
                {status.text}
              </span>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="pt-0">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <span className="text-muted-foreground">
                Total Predictions: <span className="font-medium">{predictions.length}</span>
              </span>
              {predictions.length > 0 && (
                <span className="text-muted-foreground">
                  Latest: {formatRelativeTime(predictions[0]?.timestamp)}
                </span>
              )}
            </div>
            {!autoScroll && (
              <Badge variant="outline" className="text-xs">
                Auto-scroll paused
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Feed Content */}
      <Card>
        <CardContent className="p-0">
          {!isConnected && (
            <Alert variant="destructive" className="m-4 mb-0">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                WebSocket connection lost. Reconnect to see live predictions.
              </AlertDescription>
            </Alert>
          )}

          {predictions.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">No predictions yet</h3>
              <p className="text-sm">
                {isConnected 
                  ? 'Make a prediction to see it appear here in real-time'
                  : 'Connect to WebSocket to see live prediction updates'
                }
              </p>
            </div>
          ) : (
            <div 
              ref={feedRef}
              onScroll={handleScroll}
              className="max-h-96 overflow-y-auto p-4 space-y-3"
            >
              {predictions.map((prediction, index) => (
                <PredictionItem
                  key={prediction.request_id || index}
                  prediction={prediction}
                  index={index}
                  previousPrediction={predictions[index + 1]}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Feed Statistics */}
      {predictions.length > 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <DollarSign className="h-4 w-4 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Average Value</div>
                  <div className="text-lg font-bold">
                    ${formatNumber(
                      predictions.reduce((sum, p) => sum + p.prediction, 0) / predictions.length / 100000,
                      1
                    )}K
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Avg Processing</div>
                  <div className="text-lg font-bold">
                    {formatDuration(
                      predictions.reduce((sum, p) => sum + p.processing_time_ms, 0) / predictions.length
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
                <div>
                  <div className="text-sm font-medium">Highest Value</div>
                  <div className="text-lg font-bold">
                    ${formatNumber(
                      Math.max(...predictions.map(p => p.prediction)) / 100000,
                      1
                    )}K
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}