#ifndef poplibs_support_ExternalCodelet_hpp
#define poplibs_support_ExternalCodelet_hpp

#if defined(__IPU__) && !defined(POPLIBS_DISABLE_ASM_CODELETS)
#define IS_EXTERNAL_CODELET(pred) static const bool isExternalCodelet = pred
#else
#define IS_EXTERNAL_CODELET(pred) static const bool isExternalCodelet = false
#endif

#endif // poplibs_support_ExternalCodelet_hpp
