/**
 * ErrorLogDisplay Component
 * 
 * Displays error logs with filtering, search, and real-time updates
 */

'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Search, 
  Filter, 
  RefreshCw, 
  AlertTriangle, 
  XCircle, 
  Info,
  ChevronDown,
  ChevronRight,
  Copy,
  Download
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { apiClient } from '@/services/api';
import type { ErrorLog } from '@/types';

interface ErrorLogDisplayProps {
  className?: string;
  maxHeight?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function ErrorLogDisplay({ 
  className = '', 
  maxHeight = '600px',
  autoRefresh = true,
  refreshInterval = 10000 
}: ErrorLogDisplayProps) {
  const { isConnected } = useWebSocket();
  
  // State
  const [errorLogs, setErrorLogs] = useState<ErrorLog[]>([]);
  const [filteredLogs, setFilteredLogs] = useState<ErrorLog[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());
  
  // Filter state
  const [filters, setFilters] = useState({
    level: 'all',
    search: '',
    limit: 100,
    startDate: '',
    endDate: ''
  });

  // Fetch error logs
  const fetchErrorLogs = async () => {
    setIsLoading(true);
    try {
      const logs = await apiClient.getErrorLogs({
        limit: filters.limit,
        level: filters.level === 'all' ? undefined : filters.level as any,
        search: filters.search || undefined,
        start_date: filters.startDate || undefined,
        end_date: filters.endDate || undefined
      });
      setErrorLogs(logs);
    } catch (error) {
      console.error('Failed to fetch error logs:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchErrorLogs();

    if (autoRefresh) {
      const interval = setInterval(fetchErrorLogs, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, filters]);

  // Filter logs based on search and level
  useEffect(() => {
    let filtered = errorLogs;

    // Apply search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(searchLower) ||
        log.error_type?.toLowerCase().includes(searchLower) ||
        log.endpoint?.toLowerCase().includes(searchLower)
      );
    }

    setFilteredLogs(filtered);
  }, [errorLogs, filters.search]);

  // Helper functions
  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'error': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'info': return <Info className="h-4 w-4 text-blue-500" />;
      default: return <Info className="h-4 w-4 text-gray-500" />;
    }
  };

  const getLevelBadgeVariant = (level: string) => {
    switch (level) {
      case 'error': return 'destructive';
      case 'warning': return 'secondary';
      case 'info': return 'outline';
      default: return 'outline';
    }
  };

  const toggleLogExpansion = (logId: string) => {
    const newExpanded = new Set(expandedLogs);
    if (newExpanded.has(logId)) {
      newExpanded.delete(logId);
    } else {
      newExpanded.add(logId);
    }
    setExpandedLogs(newExpanded);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const exportLogs = () => {
    const logData = filteredLogs.map(log => ({
      timestamp: log.timestamp,
      level: log.level,
      message: log.message,
      endpoint: log.endpoint,
      error_type: log.error_type,
      request_id: log.request_id
    }));

    const blob = new Blob([JSON.stringify(logData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `error_logs_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Filters */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Error Log Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search logs..."
                value={filters.search}
                onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                className="pl-10"
              />
            </div>

            {/* Level Filter */}
            <Select
              value={filters.level}
              onValueChange={(value) => setFilters(prev => ({ ...prev, level: value }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Levels</SelectItem>
                <SelectItem value="error">Error</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>

            {/* Limit */}
            <Select
              value={filters.limit.toString()}
              onValueChange={(value) => setFilters(prev => ({ ...prev, limit: parseInt(value) }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="50">50 logs</SelectItem>
                <SelectItem value="100">100 logs</SelectItem>
                <SelectItem value="200">200 logs</SelectItem>
                <SelectItem value="500">500 logs</SelectItem>
              </SelectContent>
            </Select>

            {/* Actions */}
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={fetchErrorLogs}
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={exportLogs}
                disabled={filteredLogs.length === 0}
              >
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Logs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Error Logs</span>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <span>{filteredLogs.length} logs</span>
              {!isConnected && (
                <Badge variant="outline" className="text-yellow-600">
                  Offline
                </Badge>
              )}
            </div>
          </CardTitle>
          <CardDescription>
            System error logs with filtering and search capabilities
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea style={{ height: maxHeight }}>
            <div className="p-6 space-y-3">
              {filteredLogs.map((log) => (
                <div key={log.id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                  {/* Log Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      {getLevelIcon(log.level)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant={getLevelBadgeVariant(log.level)}>
                            {log.level.toUpperCase()}
                          </Badge>
                          {log.error_type && (
                            <Badge variant="outline" className="text-xs">
                              {log.error_type}
                            </Badge>
                          )}
                          <span className="text-xs text-gray-500">
                            {new Date(log.timestamp).toLocaleString()}
                          </span>
                        </div>
                        <div className="font-medium text-sm break-words">
                          {log.message}
                        </div>
                        {log.endpoint && (
                          <div className="text-xs text-gray-600 mt-1">
                            Endpoint: <code className="bg-gray-100 px-1 rounded">{log.endpoint}</code>
                          </div>
                        )}
                        {log.request_id && (
                          <div className="text-xs text-gray-600">
                            Request ID: <code className="bg-gray-100 px-1 rounded">{log.request_id}</code>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => copyToClipboard(log.message)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                      {log.stack_trace && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleLogExpansion(log.id)}
                        >
                          {expandedLogs.has(log.id) ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* Expanded Stack Trace */}
                  {expandedLogs.has(log.id) && log.stack_trace && (
                    <div className="mt-3 pt-3 border-t">
                      <div className="text-xs text-gray-600 mb-2">Stack Trace:</div>
                      <pre className="text-xs bg-gray-100 p-3 rounded overflow-x-auto whitespace-pre-wrap">
                        {log.stack_trace}
                      </pre>
                    </div>
                  )}
                </div>
              ))}

              {filteredLogs.length === 0 && !isLoading && (
                <div className="text-center py-8 text-gray-500">
                  <Info className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>No error logs found</p>
                  <p className="text-sm">Try adjusting your filters or check back later</p>
                </div>
              )}

              {isLoading && (
                <div className="text-center py-8">
                  <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-400" />
                  <p className="text-gray-500">Loading error logs...</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}