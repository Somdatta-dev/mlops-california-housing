/**
 * Prediction Trends Component
 * 
 * Visualizes prediction trends and patterns over time including
 * volume trends, success rates, processing times, and model usage.
 */

'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Clock, 
  RefreshCw,
  BarChart3,
  PieChart as PieChartIcon
} from 'lucide-react';
import { format } from 'date-fns';

import { DatabaseTrendsResponse } from '@/types';

interface PredictionTrendsProps {
  data: DatabaseTrendsResponse | null;
  loading: boolean;
  onRefresh: () => void;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export function PredictionTrends({ data, loading, onRefresh }: PredictionTrendsProps) {
  const [activeChart, setActiveChart] = useState<'volume' | 'success' | 'performance' | 'models'>('volume');

  const formatTimestamp = (timestamp: string) => {
    try {
      return format(new Date(timestamp), 'MMM dd HH:mm');
    } catch {
      return timestamp;
    }
  };

  const formatTooltipValue = (value: number, name: string) => {
    if (name.includes('rate') || name.includes('Rate')) {
      return [`${value.toFixed(1)}%`, name];
    }
    if (name.includes('time') || name.includes('Time')) {
      return [`${value.toFixed(1)}ms`, name];
    }
    return [value.toLocaleString(), name];
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-10 w-24" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-80 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data || !data.trends) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <div className="text-center">
            <div className="text-gray-400 mb-4">
              <BarChart3 className="h-12 w-12 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No trend data available</h3>
            <p className="text-gray-500 mb-4">
              Trend data will appear here once predictions are made.
            </p>
            <Button onClick={onRefresh} variant="outline">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { trends } = data;

  // Calculate summary statistics
  const totalPredictions = trends.volume_trends.reduce((sum, item) => sum + item.total_predictions, 0);
  const avgSuccessRate = trends.success_rate_trends.length > 0 
    ? trends.success_rate_trends.reduce((sum, item) => sum + item.success_rate, 0) / trends.success_rate_trends.length
    : 0;
  const avgProcessingTime = trends.processing_time_trends.length > 0
    ? trends.processing_time_trends.reduce((sum, item) => sum + item.avg_processing_time_ms, 0) / trends.processing_time_trends.length
    : 0;
  const topModel = trends.model_usage.length > 0 ? trends.model_usage[0] : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Prediction Trends</h3>
          <p className="text-sm text-gray-600">
            {data.date_range.days} days of data ({format(new Date(data.date_range.start_date), 'MMM dd')} - {format(new Date(data.date_range.end_date), 'MMM dd')})
          </p>
        </div>
        <Button onClick={onRefresh} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Volume</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalPredictions.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Predictions in period
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Success Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{avgSuccessRate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              Average success rate
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{avgProcessingTime.toFixed(0)}ms</div>
            <p className="text-xs text-muted-foreground">
              Average response time
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Model</CardTitle>
            <PieChartIcon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{topModel?.model_version || 'N/A'}</div>
            <p className="text-xs text-muted-foreground">
              {topModel?.usage_count.toLocaleString() || 0} predictions
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Chart Selection */}
      <div className="flex space-x-2">
        <Button
          variant={activeChart === 'volume' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setActiveChart('volume')}
        >
          Volume Trends
        </Button>
        <Button
          variant={activeChart === 'success' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setActiveChart('success')}
        >
          Success Rate
        </Button>
        <Button
          variant={activeChart === 'performance' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setActiveChart('performance')}
        >
          Performance
        </Button>
        <Button
          variant={activeChart === 'models' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setActiveChart('models')}
        >
          Model Usage
        </Button>
      </div>

      {/* Charts */}
      <Card>
        <CardHeader>
          <CardTitle>
            {activeChart === 'volume' && 'Prediction Volume Over Time'}
            {activeChart === 'success' && 'Success Rate Over Time'}
            {activeChart === 'performance' && 'Processing Time Over Time'}
            {activeChart === 'models' && 'Model Usage Distribution'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            {activeChart === 'volume' && (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends.volume_trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTimestamp}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(label) => formatTimestamp(label)}
                    formatter={formatTooltipValue}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="total_predictions"
                    stackId="1"
                    stroke="#8884d8"
                    fill="#8884d8"
                    name="Total Predictions"
                  />
                  <Area
                    type="monotone"
                    dataKey="successful_predictions"
                    stackId="2"
                    stroke="#82ca9d"
                    fill="#82ca9d"
                    name="Successful"
                  />
                  <Area
                    type="monotone"
                    dataKey="failed_predictions"
                    stackId="2"
                    stroke="#ffc658"
                    fill="#ffc658"
                    name="Failed"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}

            {activeChart === 'success' && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends.success_rate_trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTimestamp}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    labelFormatter={(label) => formatTimestamp(label)}
                    formatter={formatTooltipValue}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="success_rate"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    name="Success Rate (%)"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}

            {activeChart === 'performance' && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends.processing_time_trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTimestamp}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(label) => formatTimestamp(label)}
                    formatter={formatTooltipValue}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="avg_processing_time_ms"
                    stroke="#8884d8"
                    strokeWidth={2}
                    name="Avg Processing Time (ms)"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}

            {activeChart === 'models' && (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={trends.model_usage}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ model_version, percent }) => `${model_version} (${((percent || 0) * 100).toFixed(0)}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="usage_count"
                  >
                    {trends.model_usage.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value.toLocaleString(), 'Predictions']} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}