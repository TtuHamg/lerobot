# Franka closed-loop staged validation runbook

This runbook adds orchestration, read-only health evidence, and offline
fault-injection artifacts. It does not change the gateway, controller, ROS
interfaces, launch files, or the approved async plan.

## Safety invariant

The orchestrator never calls `set_armed(true)`. A successful `start` means only
that the reviewed processes started and their readiness probes passed. It does
not mean that the gateway is armed, the controller is active, or the robot can
move.

The three accepted modes are:

| Mode | Gateway contract | Policy client | Actuation |
|---|---|---|---|
| `shadow` | `enabled=true`, `shadow=true`, 50 points | may be configured | impossible through gateway |
| `preflight-only` | `preflight_only=true`, 50 points | allowed for real model chunks | command publication is impossible |
| `isolated-single-waypoint` | isolated validation topic, max 1 | full-model client forbidden | fake/isolated endpoint only; never auto-arms |

`isolated-single-waypoint` rejects configured `arm`, `hardware_controller`,
`client`, and `policy_server` roles. It is not a shortcut to a real-arm test.

## 1. Review the process manifest

Start from:

```text
/home/pnp/franka/scripts/closed_loop_processes.example.json
```

Commands are JSON argv arrays, not shell snippets. Add site-specific `arm_state`,
`moveit`, `client`, `tunnel`, or `fake_controller` entries only after reviewing
their exact argv and readiness probe. Run from a terminal in which the ROS
workspaces are already sourced. Do not put Cookie, token, password, credential,
or private-key fields in the manifest; the validator rejects secret-like keys.

Each process needs a role-specific `match` string. Before start, all matching OS
commands are counted; any collision aborts the complete start. The state file
records PIDs and exact commands so stop only signals owned process groups.

The example deliberately omits a full-model client, MoveIt scene owner, arm
driver, and controller command. Those are deployment-specific and must not be
guessed by a generic script.

## 2. Start, status, and stop

```bash
python /home/pnp/franka/scripts/franka_closed_loop.py start \
  --mode shadow \
  --config /home/pnp/franka/scripts/closed_loop_processes.example.json

python /home/pnp/franka/scripts/franka_closed_loop.py status \
  --config /home/pnp/franka/scripts/closed_loop_processes.example.json

python /home/pnp/franka/scripts/franka_closed_loop.py stop
```

Start is transactional: if a configured process exits or its readiness probe
times out, processes started in that invocation are stopped in reverse order.
Stop does not use broad `pkill` patterns and leaves unrelated processes alone.

## 3. Capture read-only realtime health

```bash
python /home/pnp/franka/scripts/franka_realtime_health.py \
  --expected-mode shadow \
  --artifact /tmp/franka_closed_loop/health-shadow.json
```

The probe reads only OS and ROS state. It checks:

- duplicate arm/gateway/MoveIt/client/camera/tunnel processes;
- 1/5/15-minute load and highest CPU consumers;
- lifecycle state of `joint_impedance_ik_controller`;
- source-stamp age for EEF pose and arm joint state;
- textual FCI communication/error/reflex indicators;
- gateway `enabled`, `shadow`, `max_waypoints`, and chunk topic;
- gateway status freshness and unexpected `armed=true`;
- local port 8080 and client/tunnel established connections.

Exit code is `0` for PASS, `1` when only WARN exists, and `2` for FAIL. Missing
ROS topics are FAIL because freshness cannot be inferred from process presence.
The FCI check is supplementary: absence of a textual error is not a robot safety
certificate.

## 4. Generate hardware-free staged artifacts

```bash
python \
  /home/pnp/Projects/lerobot/franka_project/scripts/closed_loop_validation_probe.py \
  --artifact /tmp/franka_closed_loop/validation-offline.json
```

The JSON artifact covers:

1. fake/full 50-point shape and finite contract;
2. stale feedback producing HOLD;
3. expired chunk rejection;
4. replay and sequence/session restart around an active horizon;
5. accepted 2- and 5-waypoint short chunks.

This is an independent fault model, not proof that the deployed C++ gateway
behaves identically. Use it as the first gate, then obtain SHADOW ROS ACK
evidence from the real gateway. The probe imports no ROS module and cannot
publish or actuate.

## 5. Staged acceptance checklist

### Gate A — offline

- [ ] standalone tests pass;
- [ ] validation artifact has `overall=PASS`;
- [ ] artifact has `hardware_actuation=false`;
- [ ] 50-, 2-, and 5-point scenarios are present;
- [ ] expired, replay, restart, and timeout/HOLD cases match expected results.

### Gate B — SHADOW

- [ ] process status reports one gateway and no duplicate role;
- [ ] gateway health matches `shadow`;
- [ ] EEF and joint source stamps are fresh;
- [ ] full-model 50-point contract is preserved server-side;
- [ ] gateway ACK is `ACCEPTED_SHADOW` for fresh 2–50 point suffixes;
- [ ] expired/replayed chunks receive the documented reject ACK;
- [ ] `/franka/safe_joint_command` has no data;
- [ ] gateway status is fresh and `armed=false`.

### Gate C — preflight-only

- [ ] the PolicyServer/LeRobot client may supply real model chunks, while
  `preflight_only=true` prevents every `SafeJointCommand` publication;
- [ ] gateway starts unarmed in HOLD;
- [ ] controller lifecycle state is explicitly recorded;
- [ ] monitored world, state freshness, and preflight availability are recorded;
- [ ] timeout removes any candidate state and remains HOLD;
- [ ] no command publication is observed.

### Gate D — isolated fake single waypoint

- [ ] max waypoint count is 1 and topic is
  `/lerobot/franka/validation_action_chunk`;
- [ ] no arm/hardware-controller/full-model process is in the manifest;
- [ ] fake endpoint validates deadline and monotonic sequence behavior;
- [ ] restart begins a new session only after the prior active horizon;
- [ ] timeout causes fake-controller HOLD;
- [ ] artifact and logs identify fake hardware unambiguously.

### Gate E — real hardware review

Stop here. Real-arm arming, deadman/E-stop verification, low-stiffness setup,
controller-applied feedback, robot-mode/error integration, and gripper execution
remain separate review gates. A 50-point model chunk must never be armed
automatically.

## 6. Artifact retention

Keep one directory per attempt:

```text
<run>/
  health-<mode>.json
  validation-offline.json
  logs/
  operator-notes.md
```

Record host, git revisions, ROS domain, mode, start/stop times, and all WARN/FAIL
dispositions. Artifacts prove observations at their timestamps only; they are
not reusable readiness authorization for a later run.
