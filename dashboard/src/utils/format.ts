/**
 * Utility functions for formatting data
 */

import { format, formatDistanceToNow, parseISO } from 'date-fns';

// Format numbers
export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

export function formatPercentage(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

// Format dates
export function formatDate(date: string | Date): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'MMM dd, yyyy HH:mm:ss');
}

export function formatRelativeTime(date: string | Date): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return formatDistanceToNow(dateObj, { addSuffix: true });
}

// Format model metrics
export function formatMetric(value: number, type: 'rmse' | 'mae' | 'r2' | 'accuracy'): string {
  switch (type) {
    case 'rmse':
    case 'mae':
      return formatNumber(value, 4);
    case 'r2':
    case 'accuracy':
      return formatPercentage(value, 2);
    default:
      return formatNumber(value, 3);
  }
}

// Format GPU metrics
export function formatGPUUtilization(value: number): string {
  return `${value.toFixed(0)}%`;
}

export function formatGPUMemory(used: number, total: number): string {
  const usedGB = used / (1024 ** 3);
  const totalGB = total / (1024 ** 3);
  const percentage = (used / total) * 100;
  
  return `${usedGB.toFixed(1)}GB / ${totalGB.toFixed(1)}GB (${percentage.toFixed(0)}%)`;
}

export function formatTemperature(celsius: number): string {
  return `${celsius.toFixed(0)}°C`;
}

export function formatPower(watts: number): string {
  return `${(watts / 1000).toFixed(1)}W`;
}

// Format status badges
export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'idle':
      return 'bg-gray-100 text-gray-800';
    case 'training':
      return 'bg-blue-100 text-blue-800';
    case 'completed':
      return 'bg-green-100 text-green-800';
    case 'error':
    case 'failed':
      return 'bg-red-100 text-red-800';
    case 'paused':
      return 'bg-yellow-100 text-yellow-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
}

// Format health status
export function getHealthStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'healthy':
    case 'ok':
      return 'text-green-600';
    case 'warning':
      return 'text-yellow-600';
    case 'error':
    case 'critical':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
}

// Validation helpers
export function isValidNumber(value: unknown): boolean {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}