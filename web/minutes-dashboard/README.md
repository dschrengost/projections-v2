# Minutes Dashboard

## Deployment Services

### Production Dashboard (Port 8501)
- **Service**: `minutes-dashboard.service`
- **Port**: 8501
- **Working Directory**: `/home/daniel/prod/projections-v2`
- **Data Root**: `/home/daniel/projections-data`

### Development Dashboard (Port 8502)
This is a parallel instance for testing baseline-only LGBM minutes without the rotation SetTransformer overlay.

- **Service**: `minutes-dashboard-dev.service`
- **Port**: 8502
- **Working Directory**: `/home/daniel/projects/projections-v2` (dev repo)
- **Data Root**: `/home/daniel/projections-data`

#### How to Start/Stop
```bash
# Install and start the dev service
sudo cp infra/systemd/minutes-dashboard-dev.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now minutes-dashboard-dev

# Check status
systemctl status minutes-dashboard-dev
journalctl -u minutes-dashboard-dev -n 100 --no-pager

# Stop the dev service
sudo systemctl stop minutes-dashboard-dev
```

#### Verifying Baseline-Only Minutes
1. Check Prefect logs for: `[rotation_minutes] rotation_set_minutes disabled; using baseline minutes_v1 only`
2. Access the dev dashboard at: `http://localhost:8502`
3. Compare minute projections with the prod dashboard at: `http://localhost:8501`

---

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
