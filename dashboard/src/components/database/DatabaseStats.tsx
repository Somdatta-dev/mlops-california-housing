/**
 * Database Stats Component
 * 
 * Displays detailed statistics and performance metrics
 * with filtering capabilities and comparative analysis.
 */

'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  Database, 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  CheckCircle, 
  XCircle, 
  RefreshCw,
  Calendar,
  Filter
} from 'lucide-react';

import { DatabaseStatsResponse, FilterConfig } from '@/types';

interface DatabaseStatsProps {
  data: DatabaseStatsResponse | null;
  loading: boolean;
  onRefresh: () => void;
  filters: FilterConfig;
  onFiltersChange: (filters: FilterConfig) => void;
}

const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6'];

export function DatabaseStats({ data, loading, onRefresh, filters, onFiltersChange }: DatabaseStatsProps) {
  const [selectedPeriod, setSelectedPeriod] = useState<'7d' | '30d' | '90d' | 'all'>('30d');

  const handlePeriodChange = (period: '7d' | '30d' | '90d' | 'all') => {
    setSelectedPeriod(period);
    
    const now = new Date();
    let startDate: string | undefined;
    
    switch (period) {
      case '7d':
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
        break;
      case '30d':
        startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();
        break;
      case '90d':
        startDate = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000).toISOString();
        break;
      case 'all':
        startDate = undefined;
        break;
    }
    
    onFiltersChange({
      ...filters,
      dateRange: {
        start: startDate || undefined,
        end: now.toISOString(),
      },
    });
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-10 w-24" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-4 w-32" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-20" />
                <Skeleton className="h-3 w-24 mt-2" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <div className="text-center">
            <div className="text-gray-400 mb-4">
              <Database className="h-12 w-12 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No statistics available</h3>
            <p className="text-gray-500 mb-4">
              Statistics will appear here once predictions are made.
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

  // Prepare chart data
  const statusData = [
    { name: 'Successful', value: data.successful_predictions, color: '#22c55e' },
    { name: 'Failed', value: data.failed_predictions, color: '#ef4444' },
  ];

  const performanceData = [
    { 
      name: 'Success Rate', 
      value: data.success_rate, 
      target: 95,
      color: data.success_rate >= 95 ? '#22c55e' : data.success_rate >= 90 ? '#f59e0b' : '#ef4444'
    },
    { 
      name: 'Avg Response Time', 
      value: data.average_processing_time_ms, 
      target: 100,
      color: data.average_processing_time_ms <= 100 ? '#22c55e' : data.average_processing_time_ms <= 200 ? '#f59e0b' : '#ef4444'
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header with Period Selection */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Detailed Statistics</h3>
          <p className="text-sm text-gray-600">
            Comprehensive performance metrics and analysis
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            {(['7d', '30d', '90d', 'all'] as const).map((period) => (
              <Button
                key={period}
                variant={selectedPeriod === period ? 'default' : 'outline'}
                size="sm"
                onClick={() => handlePeriodChange(period)}
              >
                {period === 'all' ? 'All Time' : period.toUpperCase()}
              </Button>
            ))}
          </div>
          <Button onClick={onRefresh} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Total Predictions */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.total_predictions.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              All predictions in selected period
            </p>
            <div className="mt-2">
              <div className="flex items-center space-x-2 text-xs">
                <CheckCircle className="h-3 w-3 text-green-600" />
                <span>{data.successful_predictions.toLocaleString()} successful</span>
              </div>
              <div className="flex items-center space-x-2 text-xs">
                <XCircle className="h-3 w-3 text-red-600" />
                <span>{data.failed_predictions.toLocaleString()} failed</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Success Rate */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            {data.success_rate >= 95 ? (
              <TrendingUp className="h-4 w-4 text-green-600" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-600" />
            )}
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.success_rate.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              Prediction success rate
            </p>
            <div className="mt-2">
              <Progress 
                value={data.success_rate} 
                className="h-2"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>0%</span>
                <span>Target: 95%</span>
                <span>100%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Average Processing Time */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.average_processing_time_ms.toFixed(0)}ms</div>
            <p className="text-xs text-muted-foreground">
              Average response time
            </p>
            <div className="mt-2">
              <Badge 
                variant={data.average_processing_time_ms <= 100 ? 'default' : 'secondary'}
                className={
                  data.average_processing_time_ms <= 100 
                    ? 'bg-green-100 text-green-800' 
                    : data.average_processing_time_ms <= 200 
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
                }
              >
                {data.average_processing_time_ms <= 100 ? 'Excellent' : 
                 data.average_processing_time_ms <= 200 ? 'Good' : 'Needs Improvement'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Error Rate */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
            <XCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(100 - data.success_rate).toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              Prediction failure rate
            </p>
            <div className="mt-2">
              <div className="text-xs text-muted-foreground">
                {data.failed_predictions.toLocaleString()} failed out of {data.total_predictions.toLocaleString()} total
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Throughput */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Daily Average</CardTitle>
            <Calendar className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {selectedPeriod === 'all' ? 'N/A' : Math.round(data.total_predictions / parseInt(selectedPeriod)).toLocaleString()}
            </div>
            <p className="text-xs text-muted-foreground">
              Predictions per day
            </p>
            <div className="mt-2">
              <div className="text-xs text-muted-foreground">
                Based on {selectedPeriod === 'all' ? 'all time' : selectedPeriod} period
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Quality Score */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Quality</CardTitle>
            <Filter className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {data.success_rate >= 98 ? 'A+' : 
               data.success_rate >= 95 ? 'A' : 
               data.success_rate >= 90 ? 'B' : 
               data.success_rate >= 80 ? 'C' : 'D'}
            </div>
            <p className="text-xs text-muted-foreground">
              Overall data quality grade
            </p>
            <div className="mt-2">
              <div className="text-xs text-muted-foreground">
                Based on success rate and consistency
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Status Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} (${((percent || 0) * 100).toFixed(1)}%)`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {statusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value.toLocaleString(), 'Predictions']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>Performance vs Targets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Success Rate</span>
                  <span className="text-sm text-muted-foreground">{data.success_rate.toFixed(1)}% / 95%</span>
                </div>
                <Progress value={Math.min(data.success_rate, 100)} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Response Time</span>
                  <span className="text-sm text-muted-foreground">{data.average_processing_time_ms.toFixed(0)}ms / 100ms</span>
                </div>
                <Progress 
                  value={Math.min((100 / data.average_processing_time_ms) * 100, 100)} 
                  className="h-2" 
                />
              </div>

              <div className="pt-4 border-t">
                <h4 className="text-sm font-medium mb-2">Performance Summary</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Overall Health:</span>
                    <Badge variant={data.success_rate >= 95 && data.average_processing_time_ms <= 100 ? 'default' : 'secondary'}>
                      {data.success_rate >= 95 && data.average_processing_time_ms <= 100 ? 'Excellent' : 
                       data.success_rate >= 90 && data.average_processing_time_ms <= 200 ? 'Good' : 'Needs Attention'}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Reliability:</span>
                    <span className="text-muted-foreground">{data.success_rate.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Performance:</span>
                    <span className="text-muted-foreground">{data.average_processing_time_ms.toFixed(0)}ms avg</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}