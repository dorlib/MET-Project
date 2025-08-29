// Debug script to test scan details API call
// Run this in the browser console on the scan details page

async function testScanDetailsAPI() {
    const jobId = '22732ca7-2faa-49e4-b2e3-af36a7da82e8';
    
    try {
        console.log('Testing scan details API...');
        
        // Test direct fetch
        const response = await fetch(`http://localhost:8000/scans/${jobId}/details`);
        console.log('Response status:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('API Response:', data);
            console.log('Processing duration:', data.data?.processing_duration);
        } else {
            console.error('API Error:', response.statusText);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
}

// Run the test
testScanDetailsAPI();
