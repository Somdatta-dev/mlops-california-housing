/**
 * AlertSystem Component
 * 
 * Displays system alerts with filtering and resolution capabilities
 */

'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Info,
  Bell,
  BellOff,
  Eye,
  EyeOff,
  RefreshCw,
  Filter,
  Clock,
  Server,
  Cpu,
  Zap,
  Activity
} from 'lucide-react';
import { useWebSocket, useSystemHealth } from '@/hooks/useWebSocket';
import { apiClient } from '@/services/api';
import type { SystemAlert } from '@/types';

interface AlertSystemProps {
  className?: string;
  maxHeight?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
  showFilters?: boolean;
}

export function AlertSystem({ 
  className = '', 
  maxHeight = '500px',
  autoRefresh = true,
  refreshInterval = 30000,
  showFilters = true
}: AlertSystemProps) {
  const { isConnected } = useWebSocket();
  const { alerts: wsAlerts, clearAlerts } = useSystemHealth();
  
  // State
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [resolvingAlerts, setResolvingAlerts] = useState<Set<string>>(new Set());
  
  // Filter state
  const [filters, setFilters] = useState({
    resolved: false,
    severity: 'all',
    type: 'all'
  });

  // Fetch system alerts
  const fetchSystemAlerts = async () => {
    setIsLoading(true);
    try {
      const alerts = await apiClient.getSystemAlerts({
        resolved: filters.resolved,
        severity: filters.severity === 'all' ? undefined : filters.severity as any,
        type: filters.type === 'all' ? undefined : filters.type as any
      });
      setSystemAlerts(alerts);
    } catch (error) {
      console.error('Failed to fetch system alerts:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchSystemAlerts();

    if (autoRefresh) {
      const interval = setInterval(fetchSystemAlerts, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, filters]);

  // Helper functions
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'high': return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      case 'medium': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'low': return <Info className="h-4 w-4 text-blue-500" />;
      default: return <Info className="h-4 w-4 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-blue-100 text-blue-800 border-blue-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'system': return <Server className="h-4 w-4" />;
      case 'api': return <Activity className="h-4 w-4" />;
      case 'model': return <Cpu className="h-4 w-4" />;
      case 'gpu': return <Zap className="h-4 w-4" />;
      default: return <Info className="h-4 w-4" />;
    }
  };

  const resolveAlert = async (alertId: string) => {
    setResolvingAlerts(prev => new Set(prev).add(alertId));
    try {
      await apiClient.resolveAlert(alertId);
      await fetchSystemAlerts(); // Refresh alerts
    } catch (error) {
      console.error('Failed to resolve alert:', error);
    } finally {
      setResolvingAlerts(prev => {
        const newSet = new Set(prev);
        newSet.delete(alertId);
        return newSet;
      });
    }
  };

  const getAlertPriority = (alert: SystemAlert) => {
    const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
    return severityOrder[alert.severity as keyof typeof severityOrder] || 0;
  };

  // Sort alerts by priority and timestamp
  const sortedAlerts = [...systemAlerts].sort((a, b) => {
    if (a.resolved !== b.resolved) {
      return a.resolved ? 1 : -1; // Unresolved first
    }
    const priorityDiff = getAlertPriority(b) - getAlertPriority(a);
    if (priorityDiff !== 0) return priorityDiff;
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });

  // Count alerts by severity
  const alertCounts = systemAlerts.reduce((acc, alert) => {
    if (!alert.resolved) {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);

  const totalActiveAlerts = Object.values(alertCounts).reduce((sum, count) => sum + count, 0);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Alert Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="border-red-200">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <XCircle className="h-4 w-4 text-red-500" />
              <div>
                <div className="text-2xl font-bold text-red-600">{alertCounts.critical || 0}</div>
                <div className="text-xs text-gray-600">Critical</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-orange-200">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-orange-500" />
              <div>
                <div className="text-2xl font-bold text-orange-600">{alertCounts.high || 0}</div>
                <div className="text-xs text-gray-600">High</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-yellow-200">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-yellow-500" />
              <div>
                <div className="text-2xl font-bold text-yellow-600">{alertCounts.medium || 0}</div>
                <div className="text-xs text-gray-600">Medium</div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="border-blue-200">
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Info className="h-4 w-4 text-blue-500" />
              <div>
                <div className="text-2xl font-bold text-blue-600">{alertCounts.low || 0}</div>
                <div className="text-xs text-gray-600">Low</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      {showFilters && (
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Alert Filters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Select
                value={filters.severity}
                onValueChange={(value) => setFilters(prev => ({ ...prev, severity: value }))}
              >
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Severities</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>
              <Select
                value={filters.type}
                onValueChange={(value) => setFilters(prev => ({ ...prev, type: value }))}
              >
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                  <SelectItem value="api">API</SelectItem>
                  <SelectItem value="model">Model</SelectItem>
                  <SelectItem value="gpu">GPU</SelectItem>
                </SelectContent>
              </Select>
              <Button
                variant="outline"
                onClick={() => setFilters(prev => ({ ...prev, resolved: !prev.resolved }))}
              >
                {filters.resolved ? <EyeOff className="h-4 w-4 mr-2" /> : <Eye className="h-4 w-4 mr-2" />}
                {filters.resolved ? 'Hide Resolved' : 'Show Resolved'}
              </Button>
              <Button
                variant="outline"
                onClick={fetchSystemAlerts}
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* WebSocket Alerts */}
      {wsAlerts.length > 0 && (
        <Alert>
          <Bell className="h-4 w-4" />
          <AlertTitle>Real-time Alerts</AlertTitle>
          <AlertDescription>
            <div className="space-y-1 mt-2">
              {wsAlerts.slice(0, 3).map((alert, index) => (
                <div key={index} className="text-sm">• {alert}</div>
              ))}
              {wsAlerts.length > 3 && (
                <div className="text-sm text-gray-600">
                  ... and {wsAlerts.length - 3} more
                </div>
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={clearAlerts}
              className="mt-2"
            >
              <BellOff className="h-4 w-4 mr-2" />
              Clear Alerts
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* System Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>System Alerts</span>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <span>{totalActiveAlerts} active</span>
              {!isConnected && (
                <Badge variant="outline" className="text-yellow-600">
                  Offline
                </Badge>
              )}
            </div>
          </CardTitle>
          <CardDescription>
            System alerts and notifications requiring attention
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea style={{ height: maxHeight }}>
            <div className="p-6 space-y-3">
              {sortedAlerts.map((alert) => (
                <div 
                  key={alert.id} 
                  className={`p-4 border rounded-lg transition-colors ${
                    alert.resolved ? 'bg-gray-50 opacity-75' : 'hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      <div className="flex items-center gap-2">
                        {getSeverityIcon(alert.severity)}
                        {getTypeIcon(alert.type)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className={getSeverityColor(alert.severity)}>
                            {alert.severity.toUpperCase()}
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            {alert.type}
                          </Badge>
                          {alert.resolved && (
                            <Badge variant="outline" className="text-green-600">
                              Resolved
                            </Badge>
                          )}
                        </div>
                        <div className="font-medium text-sm mb-1">
                          {alert.title}
                        </div>
                        <div className="text-sm text-gray-600 mb-2">
                          {alert.message}
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-500">
                          <div className="flex items-center gap-1">
                            <Server className="h-3 w-3" />
                            <span>{alert.component}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            <span>{new Date(alert.timestamp).toLocaleString()}</span>
                          </div>
                          {alert.threshold_value && alert.current_value && (
                            <div>
                              Value: {alert.current_value} / {alert.threshold_value}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      {!alert.resolved && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => resolveAlert(alert.id)}
                          disabled={resolvingAlerts.has(alert.id)}
                        >
                          {resolvingAlerts.has(alert.id) ? (
                            <RefreshCw className="h-4 w-4 animate-spin" />
                          ) : (
                            <CheckCircle className="h-4 w-4" />
                          )}
                          {resolvingAlerts.has(alert.id) ? 'Resolving...' : 'Resolve'}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {sortedAlerts.length === 0 && !isLoading && (
                <div className="text-center py-8 text-gray-500">
                  <CheckCircle className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>No alerts found</p>
                  <p className="text-sm">
                    {filters.resolved 
                      ? 'No resolved alerts match your filters'
                      : 'All systems are running normally'
                    }
                  </p>
                </div>
              )}

              {isLoading && (
                <div className="text-center py-8">
                  <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-500">Loading alerts...</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}