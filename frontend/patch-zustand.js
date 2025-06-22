// This script fixes the zustand import issues in @react-three/fiber
const fs = require('fs');
const path = require('path');

try {
  // Path to the problematic module
  const fiberPath = path.join(__dirname, 'node_modules', '@react-three', 'fiber');
  
  // Check if the directory exists
  if (fs.existsSync(fiberPath)) {
    console.log('Patching @react-three/fiber to fix zustand imports...');
    
    // Fix store.ts or similar files that import zustand
    const storeFiles = [
      path.join(fiberPath, 'dist', 'declarations', 'core', 'store.d.ts'),
      path.join(fiberPath, 'dist', 'core', 'store.js')
    ];
    
    storeFiles.forEach(file => {
      if (fs.existsSync(file)) {
        let content = fs.readFileSync(file, 'utf8');
        
        // Replace problematic imports
        content = content.replace(
          /from ['"](zustand\/traditional|zustand)['"]/g, 
          'from "zustand"'
        );
        
        fs.writeFileSync(file, content);
        console.log(`Patched ${file}`);
      }
    });
    
    console.log('Patching complete!');
  } else {
    console.log('@react-three/fiber directory not found. No patching needed.');
  }
} catch (error) {
  console.error('Error patching files:', error);
  process.exit(1);
}
