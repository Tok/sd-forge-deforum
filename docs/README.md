# Deforum Documentation

Welcome to the comprehensive documentation for Deforum, a modern functional programming-based AI animation framework.

## 📚 Documentation Structure

### [User Guide](user-guide/)
- [Getting Started](user-guide/getting-started.md) - Quick start guide for new users
- [Animation Modes](user-guide/animation-modes.md) - Complete guide to 2D, 3D, and interpolation modes
- [Prompt Enhancement](user-guide/prompt-enhancement.md) - Using WAN for advanced prompting
- [Video Generation](user-guide/video-generation.md) - Creating and exporting animations

### [Development](development/)
- [Functional Architecture](development/functional-architecture.md) - Overview of the functional programming approach
- [Testing Guide](development/testing-guide.md) - Running and writing tests
- [Contributing](development/contributing.md) - How to contribute to the project
- [Cleanup History](development/cleanup-history.md) - Historical record of major refactoring work
- [Experimental Features](development/experimental-features.md) - Cutting-edge functionality in development

### [Migration](migration/)
- [Settings Migration](migration/settings.md) - Updating settings from older versions
- [Functional Refactor](migration/functional-refactor.md) - Migrating from procedural to functional approach
- [Naming Changes](migration/naming-changes.md) - Complete list of renamed files, functions, and variables

### [UI Components](ui/)
- [Tab Structure](ui/tab-structure.md) - Overview of the user interface organization
- [Component Guide](ui/component-guide.md) - Detailed UI component documentation

### [WAN (Wan Video)](wan/)
- [WAN Overview](wan/README.md) - Introduction to WAN video functionality
- [Technical Details](wan/technical.md) - Technical implementation details
- [Fixes & Updates](wan/fixes.md) - Bug fixes and resolution notes
- [Enhancements](wan/enhancements.md) - Feature enhancements and improvements

### [API Reference](api/)
- [Data Models](api/data-models.md) - Immutable data structures and validation
- [Schedule System](api/schedules.md) - Animation schedule management
- [UI Components](api/ui-components.md) - User interface component APIs

## 🎯 Quick Navigation

### For Users
- **New to Deforum?** Start with [Getting Started](user-guide/getting-started.md)
- **Upgrading?** Check [Settings Migration](migration/settings.md)
- **Want advanced features?** See [Prompt Enhancement](user-guide/prompt-enhancement.md)

### For Developers
- **Contributing code?** Read [Contributing](development/contributing.md)
- **Understanding architecture?** See [Functional Architecture](development/functional-architecture.md)
- **Running tests?** Check [Testing Guide](development/testing-guide.md)

### For System Architects
- **Refactoring progress?** See [Cleanup History](development/cleanup-history.md)
- **Breaking changes?** Check [Migration](migration/) guides
- **Experimental features?** Review [Experimental Features](development/experimental-features.md)

## 🔧 Functional Programming Approach

Deforum has undergone a comprehensive refactoring to embrace functional programming principles:

- **Immutable Data Structures**: All configuration and state objects are immutable
- **Pure Functions**: Business logic is side-effect free and predictable
- **Type Safety**: Comprehensive type hints and validation throughout
- **Comprehensive Testing**: 190+ unit tests with high coverage
- **Modular Architecture**: Clear separation of concerns with small, focused modules

## 📈 Project Status

- **🟢 Stable**: Core animation and generation functionality
- **🟡 Active Development**: UI system modularization and aggressive renaming
- **🔵 Experimental**: Advanced WAN features and prompt enhancement
- **🟠 Planned**: Rendering system functional refactoring

## 🤝 Community & Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions for help and ideas
- **Contributing**: See [Contributing Guide](development/contributing.md) for development setup
- **Documentation**: Help improve these docs by submitting PRs

---

*This documentation reflects the ongoing functional refactoring of Deforum. Some sections may reference both legacy and modern approaches during the transition period.* 