/**
 * Header component for the dashboard
 */

'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Activity, 
  Wifi, 
  WifiOff, 
  Settings, 
  Bell,
  User
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useHealthStatus } from '@/hooks/useApi';
import { getHealthStatusColor } from '@/utils/format';

export function Header() {
  const { isConnected, connectionState } = useWebSocket();
  const { data: healthStatus, loading: healthLoading } = useHealthStatus();
  const [showNotifications, setShowNotifications] = useState(false);

  return (
    <header className="border-b bg-white px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo and Title */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Activity className="h-8 w-8 text-blue-600" />
            <h1 className="text-2xl font-bold text-gray-900">
              MLOps Dashboard
            </h1>
          </div>
          
          {/* System Status */}
          <div className="flex items-center space-x-2">
            <Badge 
              variant="outline" 
              className={`${getHealthStatusColor(healthStatus?.status || 'unknown')}`}
            >
              {healthLoading ? 'Checking...' : healthStatus?.status || 'Unknown'}
            </Badge>
            
            {healthStatus?.gpu_available && (
              <Badge variant="outline" className="text-green-600">
                GPU Available
              </Badge>
            )}
          </div>
        </div>

        {/* Right side controls */}
        <div className="flex items-center space-x-4">
          {/* WebSocket Connection Status */}
          <div className="flex items-center space-x-2">
            {isConnected ? (
              <Wifi className="h-4 w-4 text-green-600" />
            ) : (
              <WifiOff className="h-4 w-4 text-red-600" />
            )}
            <span className="text-sm text-gray-600 capitalize">
              {connectionState}
            </span>
          </div>

          {/* Notifications */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowNotifications(!showNotifications)}
            className="relative"
          >
            <Bell className="h-4 w-4" />
            <Badge 
              variant="destructive" 
              className="absolute -top-1 -right-1 h-5 w-5 rounded-full p-0 text-xs"
            >
              3
            </Badge>
          </Button>

          {/* Settings */}
          <Button variant="ghost" size="sm">
            <Settings className="h-4 w-4" />
          </Button>

          {/* User Profile */}
          <Button variant="ghost" size="sm">
            <User className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Model Info Bar */}
      {healthStatus?.model_loaded && (
        <div className="mt-3 flex items-center justify-between rounded-lg bg-blue-50 px-4 py-2">
          <div className="flex items-center space-x-4 text-sm">
            <span className="font-medium text-blue-900">
              Model Loaded: {healthStatus.version}
            </span>
            <span className="text-blue-700">
              Uptime: {Math.floor(healthStatus.uptime / 3600)}h {Math.floor((healthStatus.uptime % 3600) / 60)}m
            </span>
          </div>
          
          <Badge variant="secondary" className="bg-blue-100 text-blue-800">
            Ready for Predictions
          </Badge>
        </div>
      )}
    </header>
  );
}