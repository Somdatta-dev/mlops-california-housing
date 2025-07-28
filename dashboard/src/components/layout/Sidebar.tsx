/**
 * Sidebar navigation component
 */

'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Home,
  Brain,
  Activity,
  Database,
  Settings,
  BarChart3,
  Zap,
  Monitor,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { NavigationItem } from '@/types';

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    href: '/',
    icon: Home,
  },
  {
    id: 'predictions',
    label: 'Predictions',
    href: '/predictions',
    icon: Brain,
  },
  {
    id: 'training',
    label: 'Training',
    href: '/training',
    icon: Zap,
  },
  {
    id: 'monitoring',
    label: 'Monitoring',
    href: '/monitoring',
    icon: Activity,
  },
  {
    id: 'database',
    label: 'Database Explorer',
    href: '/database',
    icon: Database,
  },
  {
    id: 'analytics',
    label: 'Analytics',
    href: '/analytics',
    icon: BarChart3,
  },
  {
    id: 'system',
    label: 'System Health',
    href: '/system',
    icon: Monitor,
  },
];

const settingsItems: NavigationItem[] = [
  {
    id: 'settings',
    label: 'Settings',
    href: '/settings',
    icon: Settings,
  },
];

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  const NavItem = ({ item }: { item: NavigationItem }) => {
    const isActive = pathname === item.href;
    const Icon = item.icon;

    return (
      <Link href={item.href}>
        <Button
          variant={isActive ? 'secondary' : 'ghost'}
          className={cn(
            'w-full justify-start',
            collapsed && 'px-2',
            isActive && 'bg-blue-100 text-blue-900 hover:bg-blue-200'
          )}
        >
          <Icon className={cn('h-4 w-4', !collapsed && 'mr-2')} />
          {!collapsed && (
            <>
              <span>{item.label}</span>
              {item.badge && (
                <Badge variant="secondary" className="ml-auto">
                  {item.badge}
                </Badge>
              )}
            </>
          )}
        </Button>
      </Link>
    );
  };

  return (
    <div
      className={cn(
        'flex h-full flex-col border-r bg-white transition-all duration-300',
        collapsed ? 'w-16' : 'w-64',
        className
      )}
    >
      {/* Collapse Toggle */}
      <div className="flex items-center justify-end p-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setCollapsed(!collapsed)}
          className="h-8 w-8 p-0"
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3">
        {/* Main Navigation */}
        <div className="space-y-1">
          {!collapsed && (
            <h3 className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Main
            </h3>
          )}
          {navigationItems.map((item) => (
            <NavItem key={item.id} item={item} />
          ))}
        </div>

        <Separator className="my-4" />

        {/* Settings */}
        <div className="space-y-1">
          {!collapsed && (
            <h3 className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Settings
            </h3>
          )}
          {settingsItems.map((item) => (
            <NavItem key={item.id} item={item} />
          ))}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4">
        {!collapsed && (
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="text-xs text-gray-600">
              <div className="font-medium">MLOps Platform</div>
              <div>Version 1.0.0</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}