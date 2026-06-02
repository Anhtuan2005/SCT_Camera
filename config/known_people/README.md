# Known People References

Place reference images for family members in this folder, then list them in
`config/settings.yaml` under `identity.known_persons`.

Example:

```yaml
identity:
  known_persons:
  - name: Ba
    reference_images:
    - config/known_people/ba_front.jpg
    - config/known_people/ba_side.jpg
```

Use clear full-body crops from the same camera angle when possible. The current
resolver uses OpenCV color histograms, so it is suitable for a controlled demo,
not biometric-grade face recognition.
