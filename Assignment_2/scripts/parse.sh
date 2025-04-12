#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --reservation=fri
#SBATCH --job-name=parse_stats
#SBATCH --ntasks=1
#SBATCH --time=00:02:00  # Extra time for stats calculations
#SBATCH --output=test.log

INPUT_FILE="output.log"
OUTPUT_FILE="stats.log"

# Initialize arrays to store all values
declare -a hist_values cum_values lum_values assign_values total_values

# Parse STATS lines and store values
while read -r line; do
  if [[ "$line" == STATS* ]]; then
    IFS=', ' read -r _ hist cum lum assign total <<< "$line"
    hist_values+=("$hist")
    cum_values+=("$cum")
    lum_values+=("$lum")
    assign_values+=("$assign")
    total_values+=("$total")
  fi
done < "$INPUT_FILE"

# Function to calculate mean and stddev
calculate_stats() {
  local values=("$@")
  local sum=0
  local count=${#values[@]}
  
  # First calculate the mean
  for val in "${values[@]}"; do
    sum=$(bc <<< "scale=6; $sum + $val")
  done
  local mean=$(bc <<< "scale=6; $sum / $count")
  
  # Then calculate the sum of squared differences from mean
  local sum_sq_diff=0
  for val in "${values[@]}"; do
    local diff=$(bc <<< "scale=6; $mean - $val")
    sum_sq_diff=$(bc <<< "scale=6; $sum_sq_diff + ($diff * $diff)")
  done
  
  # Compute stddev (population standard deviation)
  local variance=$(bc <<< "scale=6; $sum_sq_diff / $count")
  local stddev=$(bc <<< "scale=3; sqrt($variance)")
  local mean_rounded=$(bc <<< "scale=3; $mean")
  
  echo "$mean_rounded ± $stddev"
}

# Calculate statistics for each metric
hist_stats=$(calculate_stats "${hist_values[@]}")
cum_stats=$(calculate_stats "${cum_values[@]}")
lum_stats=$(calculate_stats "${lum_values[@]}")
assign_stats=$(calculate_stats "${assign_values[@]}")
total_stats=$(calculate_stats "${total_values[@]}")

# Generate comprehensive report
{
  echo "Metric,Mean ± StdDev (ms)"
  echo "--------------------------"
  echo "RGB→YUV + Histogram   | $hist_stats"
  echo "Cumulative Histogram  | $cum_stats"
  echo "New Luminance Calc    | $lum_stats"
  echo "Assign + YUV→RGB      | $assign_stats"
  echo "Total GPU Time        | $total_stats"
  echo ""
} > "$OUTPUT_FILE"

echo "Statistics saved to $OUTPUT_FILE"