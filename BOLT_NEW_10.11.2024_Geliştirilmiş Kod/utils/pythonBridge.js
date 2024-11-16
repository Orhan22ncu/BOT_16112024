const { PythonShell } = require('python-shell');
const logger = require('./logger');

class PythonBridge {
  static async runScript(scriptPath, args = {}) {
    try {
      const options = {
        mode: 'json',
        pythonPath: 'python',
        pythonOptions: ['-u'],
        scriptPath: './model',
        args: [JSON.stringify(args)]
      };

      const results = await PythonShell.run(scriptPath, options);
      return results[0];
    } catch (error) {
      logger.error('Python script execution failed:', error);
      throw new Error('Failed to execute Python script');
    }
  }
}

module.exports = PythonBridge;