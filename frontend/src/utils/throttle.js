/**
 * Utility function to throttle repeated function calls
 * @param {Function} func The function to throttle
 * @param {number} wait The throttle wait time in milliseconds
 * @returns {Function} A throttled function
 */
export const throttle = (func, wait) => {
  let throttled = false;
  let lastArgs = null;
  
  // Return a function that can be called repeatedly, but will only execute
  // at most once per "wait" interval
  return function(...args) {
    // If we're not currently throttled, execute the function immediately
    if (!throttled) {
      const result = func.apply(this, args);
      throttled = true;
      
      // After the wait period, reset throttled state
      setTimeout(() => {
        throttled = false;
        
        // If there was a call during the wait period, call again with latest args
        if (lastArgs) {
          const pendingArgs = lastArgs;
          lastArgs = null;
          func.apply(this, pendingArgs);
        }
      }, wait);
      
      return result;
    } else {
      // Store the most recent arguments in case we need to call again later
      lastArgs = args;
      return Promise.resolve(); // Return empty promise for API calls
    }
  };
};
