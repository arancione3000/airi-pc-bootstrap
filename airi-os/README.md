# Airi OS

Airi OS is a reproducible Debian-based live operating system dedicated to Airi-PC.

## Design

Airi OS does **not** reinvent the Linux kernel. It uses Debian 13 (Trixie) as the stable hardware and package foundation, then turns Airi-PC into a first-class system service.

Core properties:

- Debian 13 / amd64 live ISO
- XFCE desktop with Airi branding
- Airi-PC preinstalled under `/opt/airi-pc`
- Airi-PC starts automatically through systemd
- Playwright/Xvfb/Openbox tooling prepared at image-build time
- persistent Airi control-plane/auth state under `/var/lib/airi-pc`
- atomic `airi-update` with test-before-switch
- `airi-rollback` to return to the previous verified release
- `airi-status` for a quick runtime health check

The first version is deliberately a **bootable live appliance**. Disk installation can be added after the live image and recovery path have been validated on real hardware.

## Build

On Debian/Ubuntu with `live-build` and `xorriso` installed:

```bash
sudo bash airi-os/build.sh
```

The resulting ISO and checksum are written to `airi-os/out/`.

## Runtime

Airi-PC listens locally on port 9010. From the desktop:

```bash
airi-status
```

Update to the latest verified `main`:

```bash
sudo airi-update
```

Rollback:

```bash
sudo airi-rollback
```
