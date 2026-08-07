#!/bin/bash
# Configure and verify the machine-local CPU/IRQ split used by Franka FCI.
set -euo pipefail

MODE="${1:---check}"
FRANKA_NIC="${FRANKA_NIC:-enp1s0}"
FCI_CPU_LIST="10"
IRQ_CPUS=(8)

usage() {
  cat <<EOF
Usage: $0 [--check|--apply]

Environment overrides:
  FRANKA_NIC       Franka Ethernet interface (default: enp1s0)
EOF
}

cpu_in_list() {
  local target="$1"
  local list="$2"
  local field start finish
  IFS=',' read -ra fields <<<"$list"
  for field in "${fields[@]}"; do
    if [[ "$field" == *-* ]]; then
      start="${field%-*}"
      finish="${field#*-}"
      if ((target >= start && target <= finish)); then
        return 0
      fi
    elif [[ "$field" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

discover_irqs() {
  awk -v nic="$FRANKA_NIC" '$0 ~ nic {gsub(":", "", $1); print $1}' /proc/interrupts
}

read_coalesce_value() {
  local key="$1"
  ethtool -c "$FRANKA_NIC" 2>/dev/null |
    awk -v key="$key" '$1 == key ":" {print $2; exit}'
}

check_configuration() {
  local failed=false
  local isolated irq affinity rx_usecs tx_usecs governor epp power_profile
  local irqs=()

  if ! ip link show dev "$FRANKA_NIC" >/dev/null 2>&1; then
    echo "ERROR: Franka NIC does not exist: $FRANKA_NIC"
    return 1
  fi

  isolated="$(< /sys/devices/system/cpu/isolated)"
  for cpu in 8 9 10 11; do
    if ! cpu_in_list "$cpu" "$isolated"; then
      echo "ERROR: CPU$cpu is not isolated; current isolated CPUs: ${isolated:-<none>}"
      failed=true
    fi
  done

  if [[ "$(< /sys/devices/system/cpu/cpu8/topology/thread_siblings_list)" != "8-9" ||
        "$(< /sys/devices/system/cpu/cpu10/topology/thread_siblings_list)" != "10-11" ]]; then
    echo "ERROR: expected CPU8/9 and CPU10/11 to be sibling pairs on this host"
    failed=true
  fi

  mapfile -t irqs < <(discover_irqs)
  if ((${#irqs[@]} == 0)); then
    echo "ERROR: no IRQ found for $FRANKA_NIC"
    failed=true
  fi
  for irq in "${irqs[@]}"; do
    affinity="$(< "/proc/irq/$irq/smp_affinity_list")"
    if [[ "$affinity" != "8" ]]; then
      echo "ERROR: IRQ $irq ($FRANKA_NIC) affinity is $affinity, expected CPU8"
      failed=true
    fi
  done

  rx_usecs="$(read_coalesce_value rx-usecs)"
  tx_usecs="$(read_coalesce_value tx-usecs)"
  if [[ "$rx_usecs" != "0" || "$tx_usecs" != "0" ]]; then
    echo "ERROR: $FRANKA_NIC interrupt coalescing is rx-usecs=${rx_usecs:-?}, tx-usecs=${tx_usecs:-?}; expected 0/0"
    failed=true
  fi

  governor="$(< /sys/devices/system/cpu/cpu10/cpufreq/scaling_governor)"
  epp="$(< /sys/devices/system/cpu/cpu10/cpufreq/energy_performance_preference)"
  power_profile="$(powerprofilesctl get 2>/dev/null || true)"
  if [[ "$power_profile" != "performance" || "$epp" != "performance" ]]; then
    echo "ERROR: power profile/EPP is ${power_profile:-unknown}/$epp, expected performance/performance"
    failed=true
  fi

  if [[ "$failed" == "true" ]]; then
    return 1
  fi
  echo "OK: $FRANKA_NIC IRQs are on CPU8; coalescing is disabled"
  echo "OK: FCI CPUs $FCI_CPU_LIST are isolated; power profile/EPP is performance"
  echo "INFO: intel_pstate governor '$governor' is informational; EPP controls the performance preference"
}

apply_configuration() {
  local index=0
  local cpu driver irq
  local irqs=()

  if ((EUID != 0)); then
    echo "ERROR: --apply requires root"
    echo "Run: sudo $0 --apply"
    return 1
  fi
  if ! command -v ethtool >/dev/null 2>&1; then
    echo "ERROR: ethtool is required"
    return 1
  fi

  mapfile -t irqs < <(discover_irqs)
  if ((${#irqs[@]} == 0)); then
    echo "ERROR: no IRQ found for $FRANKA_NIC"
    return 1
  fi
  for irq in "${irqs[@]}"; do
    cpu="${IRQ_CPUS[index % ${#IRQ_CPUS[@]}]}"
    echo "$cpu" >"/proc/irq/$irq/smp_affinity_list"
    echo "Pinned IRQ $irq ($FRANKA_NIC) to CPU$cpu"
    ((index += 1))
  done

  driver="$(ethtool -i "$FRANKA_NIC" 2>/dev/null |
    awk '$1 == "driver:" {print $2; exit}')"
  if [[ "$driver" == "igc" ]]; then
    # igc Queue Pair mode exposes tx-usecs as a mirror of rx-usecs and rejects
    # requests that set both fields in one operation.
    ethtool -C "$FRANKA_NIC" rx-usecs 0
  else
    ethtool -C "$FRANKA_NIC" rx-usecs 0 tx-usecs 0
  fi
  ethtool --set-eee "$FRANKA_NIC" eee off
  echo "Configured $FRANKA_NIC with zero interrupt coalescing and EEE disabled"
}

case "$MODE" in
--check)
  check_configuration
  ;;
--apply)
  apply_configuration
  check_configuration
  ;;
-h | --help)
  usage
  ;;
*)
  usage
  exit 2
  ;;
esac
