/**************************************************************************
 *
 * Copyright 2008 VMware, Inc.
 * Copyright 2009-2010 Chia-I Wu <olvaffe@gmail.com>
 * Copyright 2010-2011 LunarG, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "c11/threads.h"

#include "eglglobals.h"
#include "egldisplay.h"
#include "egldriver.h"


static mtx_t _eglGlobalMutex = _MTX_INITIALIZER_NP;

struct _egl_global _eglGlobal =
{
   &_eglGlobalMutex,       /* Mutex */
   NULL,                   /* DisplayList */
   2,                      /* NumAtExitCalls */
   {
      /* default AtExitCalls, called in reverse order */
      _eglUnloadDrivers, /* always called last */
      _eglFiniDisplay
   },

   /* ClientOnlyExtensionString */
   "EGL_EXT_client_extensions"
   " EGL_EXT_platform_base"
   " EGL_KHR_client_get_all_proc_addresses"
   " EGL_KHR_debug",

   /* PlatformExtensionString */
#ifdef HAVE_WAYLAND_PLATFORM
   " EGL_EXT_platform_wayland"
#endif
#ifdef HAVE_X11_PLATFORM
   " EGL_EXT_platform_x11"
#endif
#ifdef HAVE_DRM_PLATFORM
   " EGL_MESA_platform_gbm"
#endif
#ifdef HAVE_SURFACELESS_PLATFORM
   " EGL_MESA_platform_surfaceless"
#endif
   "",

   NULL, /* ClientExtensionsString */

   NULL, /* debugCallback */
   _EGL_DEBUG_BIT_CRITICAL | _EGL_DEBUG_BIT_ERROR, /* debugTypesEnabled */
};


static void
_eglAtExit(void)
{
   EGLint i;
   for (i = _eglGlobal.NumAtExitCalls - 1; i >= 0; i--)
      _eglGlobal.AtExitCalls[i]();
}


void
_eglAddAtExitCall(void (*func)(void))
{
   if (func) {
      static EGLBoolean registered = EGL_FALSE;

      mtx_lock(_eglGlobal.Mutex);

      if (!registered) {
         atexit(_eglAtExit);
         registered = EGL_TRUE;
      }

      assert(_eglGlobal.NumAtExitCalls < ARRAY_SIZE(_eglGlobal.AtExitCalls));
      _eglGlobal.AtExitCalls[_eglGlobal.NumAtExitCalls++] = func;

      mtx_unlock(_eglGlobal.Mutex);
   }
}

const char *
_eglGetClientExtensionString(void)
{
   const char *ret;

   mtx_lock(_eglGlobal.Mutex);

   if (_eglGlobal.ClientExtensionString == NULL) {
      size_t clientLen = strlen(_eglGlobal.ClientOnlyExtensionString);
      size_t platformLen = strlen(_eglGlobal.PlatformExtensionString);

      _eglGlobal.ClientExtensionString = (char *) malloc(clientLen + platformLen + 1);
      if (_eglGlobal.ClientExtensionString != NULL) {
         char *ptr = _eglGlobal.ClientExtensionString;

         memcpy(ptr, _eglGlobal.ClientOnlyExtensionString, clientLen);
         ptr += clientLen;

         if (platformLen > 0) {
            // Note that if PlatformExtensionString is not empty, then it will
            // already have a leading space.
            assert(_eglGlobal.PlatformExtensionString[0] == ' ');
            memcpy(ptr, _eglGlobal.PlatformExtensionString, platformLen);
            ptr += platformLen;
         }
         *ptr = '\0';
      }
   }
   ret = _eglGlobal.ClientExtensionString;

   mtx_unlock(_eglGlobal.Mutex);
   return ret;
}
