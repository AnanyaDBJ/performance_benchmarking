#!/bin/bash
# Quick check to see if benchmark is actively making requests

echo "=============================================================================="
echo "  BENCHMARK ACTIVITY CHECK"
echo "=============================================================================="
echo ""

# Check if Python processes are running
echo "1. Checking for running Python processes..."
PYTHON_PROCS=$(ps aux | grep -i "src/compare.*endpoints.py\|src/benchmark_llm.py" | grep -v grep | wc -l | tr -d ' ')
if [ "$PYTHON_PROCS" -gt 0 ]; then
    echo "   ✓ Found $PYTHON_PROCS benchmark process(es) running"
    ps aux | grep -i "src/compare.*endpoints.py\|src/benchmark_llm.py" | grep -v grep | awk '{print "     PID:", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| CMD:", $11, $12, $13}'
else
    echo "   ✗ No benchmark processes found"
fi
echo ""

# Check for active network connections to Databricks
echo "2. Checking for active connections to Databricks..."
CONNECTIONS=$(netstat -an 2>/dev/null | grep -i "databricks\|443\|ESTABLISHED" | grep -v "127.0.0.1" | wc -l | tr -d ' ')
if [ "$CONNECTIONS" -gt 0 ]; then
    echo "   ✓ Found $CONNECTIONS active network connection(s)"
else
    echo "   ⚠ No active connections found (may be between requests)"
fi
echo ""

# Check recent output files
echo "3. Checking for recently created output files..."
RECENT_FILES=$(find . -name "*.png" -o -name "*.json" -mmin -10 2>/dev/null | wc -l | tr -d ' ')
if [ "$RECENT_FILES" -gt 0 ]; then
    echo "   ✓ Found $RECENT_FILES file(s) created in last 10 minutes"
    echo "   Recent files:"
    find . -name "*.png" -o -name "*.json" -mmin -10 2>/dev/null | head -5 | xargs ls -lh | awk '{print "     ", $9, "("$5")", $6, $7, $8}'
else
    echo "   ⚠ No recent output files found"
fi
echo ""

# Show recent file modifications
echo "4. Most recently modified files..."
ls -lt | head -6 | tail -5 | awk '{print "     ", $9, "-", $6, $7, $8}'
echo ""

# CPU usage check
echo "5. System activity..."
TOP_CPU=$(ps aux | sort -rk 3,3 | head -5 | grep python | head -1)
if [ -n "$TOP_CPU" ]; then
    echo "   ✓ Python process using CPU:"
    echo "     $TOP_CPU" | awk '{print "     PID:", $2, "| CPU:", $3"%"}'
else
    echo "   ⚠ No Python process using significant CPU"
fi
echo ""

echo "=============================================================================="
echo "  QUICK DIAGNOSTICS"
echo "=============================================================================="

# Provide suggestions
if [ "$PYTHON_PROCS" -eq 0 ]; then
    echo "⚠ No benchmark process running - did it complete or crash?"
    echo "  Check: cat nohup.out (if run with nohup)"
elif [ "$RECENT_FILES" -eq 0 ]; then
    echo "⚠ Process running but no recent output - might be stuck or slow"
    echo "  Consider: Check process logs or add --timeout parameter"
else
    echo "✓ Benchmark appears to be running normally"
    echo "  Processes: $PYTHON_PROCS | Recent files: $RECENT_FILES"
fi

echo ""
echo "To monitor in real-time:"
echo "  - Run: python utils/monitor_live.py --endpoints <ep1> <ep2> <ep3> <ep4>    (in another terminal)"
echo "  - Or:  tail -f nohup.out         (if running with nohup)"
echo "  - Or:  watch -n 5 'ls -lt | head -10'"
echo "=============================================================================="
