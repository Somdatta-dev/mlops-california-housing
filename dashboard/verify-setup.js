#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const requiredFiles = [
  'src/app/layout.tsx',
  'src/app/page.tsx',
  'src/components/layout/Header.tsx',
  'src/components/layout/Sidebar.tsx',
  'src/components/layout/Layout.tsx',
  'src/components/dashboard/DashboardOverview.tsx',
  'src/hooks/useApi.ts',
  'src/hooks/useWebSocket.ts',
  'src/services/api.ts',
  'src/services/websocket.ts',
  'src/types/index.ts',
  'src/utils/format.ts',
  'src/lib/utils.ts',
  '.env.local',
  'package.json',
  'README.md'
];

const requiredDirectories = [
  'src/app',
  'src/components/layout',
  'src/components/dashboard',
  'src/components/ui',
  'src/hooks',
  'src/services',
  'src/types',
  'src/utils',
  'src/lib'
];

console.log('🔍 Verifying Next.js Dashboard Setup...\n');

// Check directories
console.log('📁 Checking directories:');
requiredDirectories.forEach(dir => {
  const exists = fs.existsSync(path.join(__dirname, dir));
  console.log(`  ${exists ? '✅' : '❌'} ${dir}`);
});

console.log('\n📄 Checking files:');
requiredFiles.forEach(file => {
  const exists = fs.existsSync(path.join(__dirname, file));
  console.log(`  ${exists ? '✅' : '❌'} ${file}`);
});

// Check package.json scripts
console.log('\n📦 Checking package.json scripts:');
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
const requiredScripts = ['dev', 'build', 'start', 'lint'];
requiredScripts.forEach(script => {
  const exists = packageJson.scripts && packageJson.scripts[script];
  console.log(`  ${exists ? '✅' : '❌'} ${script}`);
});

// Check key dependencies
console.log('\n📚 Checking key dependencies:');
const requiredDeps = [
  'next',
  'react',
  'react-dom',
  'typescript',
  'tailwindcss',
  'lucide-react',
  'recharts',
  'ws',
  'date-fns'
];
requiredDeps.forEach(dep => {
  const exists = (packageJson.dependencies && packageJson.dependencies[dep]) || 
                 (packageJson.devDependencies && packageJson.devDependencies[dep]);
  console.log(`  ${exists ? '✅' : '❌'} ${dep}`);
});

console.log('\n🎉 Dashboard setup verification complete!');
console.log('\n🚀 To start the development server:');
console.log('   npm run dev');
console.log('\n🏗️  To build for production:');
console.log('   npm run build');