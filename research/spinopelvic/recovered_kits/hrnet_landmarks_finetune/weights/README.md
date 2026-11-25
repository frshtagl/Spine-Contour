# HRNet checkpoint staging

Expected file: `landmarks_hrnet.pt`

- Source: user-supplied `files (2).zip` → nested HRNet kit
- Size: 127,887,090 bytes
- SHA-256: `aa0f97a7f8dfe478b443ee6e87590185d5cd9278342538e8d4c67f7bdc243df9`
- This differs from the current fork LFS pointer at `backend/weights/hrnet_landmarks.pt` (127,888,626 bytes; SHA-256 `2cc7f791c9a0df3b84515ddb065e52ccff954b8264b4374b4cbdfe8f71870e36`).

Keep this recovered checkpoint separate from the upstream baseline. Add only through Git LFS after the repository is private. No sample images or annotations belong here.
