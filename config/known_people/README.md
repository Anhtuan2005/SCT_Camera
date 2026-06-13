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

Use clear face photos with one person per image. A front-facing photo and one or
two slightly different angles usually work better than a full-body crop.

The resolver uses InsightFace cosine similarity. The default `buffalo_l` model
is downloaded to `models/insightface` on first use. InsightFace code is MIT
licensed, but its provided pretrained models are limited to non-commercial
research use. Configure a separately licensed model for commercial deployment.
